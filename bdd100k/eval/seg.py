"""Evaluation procedures for semantic segmentation.

For dataset with `n` classes, we treat the index `n` as the ignored class.
When compute IoUs, this ignored class is considered.
However, IoU(ignored) doesn't influence mIoU.
"""

from functools import partial
from multiprocessing import Pool
from typing import List, Set, Tuple, cast

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64, NDArrayI32, NDArrayU8
from scalabel.eval.result import BaseResult
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import reorder_preds
from ..label.label import drivables, labels
from ..label.to_mask import IGNORE_LABEL


class SemSegResult(BaseResult):
    """The class for semantic segmentation evaluation results."""

    IoU: List[float]
    Acc: List[float]
    fIoU: float
    pAcc: float

    def __init__(self, *args_, **kwargs) -> None:  # type: ignore
        """Set extra parameters."""
        super().__init__(*args_, **kwargs)
        self._formatters = {
            metric: "{:.1f}".format for metric in self.__fields__
        }


def fast_hist(
    groundtruth: NDArrayU8,
    prediction: NDArrayU8,
    size: int,
) -> NDArrayI32:
    """Compute the histogram."""
    prediction = prediction.copy()
    # Out-of-range values as `ignored`
    prediction[prediction >= size] = size - 1

    k = np.logical_and(
        # `ignored` is not considered
        np.greater_equal(groundtruth, 0),
        np.less(groundtruth, size - 1),
    )
    return np.bincount(  # type: ignore
        size * groundtruth[k].astype(int) + prediction[k], minlength=size ** 2
    ).reshape(size, size)


def per_class_iou(hist: NDArrayU8) -> NDArrayF64:
    """Calculate per class iou."""
    ious: NDArrayF64 = np.diag(hist) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist)
    )
    ious[np.isnan(ious)] = 0
    # Last class as `ignored`
    return ious[:-1]  # type: ignore


def per_class_acc(hist: NDArrayU8) -> NDArrayF64:
    """Calculate per class accuracy."""
    accs: NDArrayF64 = np.diag(hist) / hist.sum(axis=0)
    accs[np.isnan(accs)] = 0
    # Last class as `ignored`
    return accs[:-1]  # type: ignore


def whole_acc(hist: NDArrayU8) -> float:
    """Calculate whole accuray."""
    hist = hist[:-1]
    return cast(float, np.diag(hist).sum() / hist.sum())


def freq_iou(hist: NDArrayU8) -> float:
    """Calculate frequency iou."""
    ious = per_class_iou(hist)
    hist = hist[:-1]
    freq = hist.sum(axis=1) / hist.sum()
    return cast(float, (ious * freq).sum())


def per_image_hist(
    gt_path: str, pred_path: str = "", num_classes: int = 2
) -> Tuple[NDArrayI32, Set[int]]:
    """Calculate per image hist."""
    assert num_classes >= 2
    assert num_classes <= IGNORE_LABEL
    gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
    gt = gt.copy()
    gt[gt == IGNORE_LABEL] = num_classes - 1
    gt_id_set = set(np.unique(gt).tolist())

    # remove `ignored`
    if num_classes - 1 in gt_id_set:
        gt_id_set.remove(num_classes - 1)

    if not pred_path:
        # Blank input feed as `ignored`
        pred = np.multiply(np.ones_like(gt, dtype=np.uint8), num_classes - 1)
    else:
        pred = np.asarray(Image.open(pred_path), dtype=np.uint8)
    hist = fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_segmentation(
    gt_paths: List[str],
    pred_paths: List[str],
    mode: str = "sem_seg",
    nproc: int = NPROC,
) -> SemSegResult:
    """Evaluate segmentation IoU from input folders."""
    assert mode in ["sem_seg", "drivable"]
    logger.info("Found %d results", len(gt_paths))
    label_defs = {
        "sem_seg": labels,
        "drivable": drivables,
    }[mode]
    categories = [label.name for label in label_defs if label.trainId != 255]
    num_classes = {
        "sem_seg": len(categories) + 1,  # add an `ignored` class
        "drivable": len(drivables),  # `background` as `ignored`
    }[mode]

    pred_paths = reorder_preds(gt_paths, pred_paths)

    if nproc > 1:
        with Pool(nproc) as pool:
            hist_and_gt_id_sets = pool.starmap(
                partial(per_image_hist, num_classes=num_classes),
                tqdm(zip(gt_paths, pred_paths), total=len(gt_paths)),
            )
    else:
        hist_and_gt_id_sets = [
            per_image_hist(gt_path, pred_path, num_classes=num_classes)
            for gt_path, pred_path in tqdm(
                zip(gt_paths, pred_paths), total=len(gt_paths)
            )
        ]
    hist = np.zeros((num_classes, num_classes))
    gt_id_set = set()
    for (hist_, gt_id_set_) in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    ious = per_class_iou(hist)
    accs = per_class_acc(hist)
    mIoU = np.mean(ious[list(gt_id_set)])  # pylint: disable=invalid-name
    mAcc = np.mean(accs[list(gt_id_set)])  # pylint: disable=invalid-name
    res_dict = dict(
        IoU=np.multiply(ious, 100).tolist() + [100 * mIoU],
        Acc=np.multiply(accs, 100).tolist() + [100 * mAcc],
        fIoU=100 * freq_iou(hist),  # pylint: disable=invalid-name
        pAcc=100 * whole_acc(hist),  # pylint: disable=invalid-name
    )

    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    return SemSegResult(
        classes=categories, hyper_classes=["AVERAGE"], **res_dict
    )


def evaluate_drivable(
    gt_paths: List[str], pred_paths: List[str], nproc: int = NPROC
) -> SemSegResult:
    """Evaluate drivable area."""
    return evaluate_segmentation(
        gt_paths, pred_paths, mode="drivable", nproc=nproc
    )


def evaluate_sem_seg(
    gt_paths: List[str], pred_paths: List[str], nproc: int = NPROC
) -> SemSegResult:
    """Evaluate semantic segmentation."""
    return evaluate_segmentation(
        gt_paths, pred_paths, mode="sem_seg", nproc=nproc
    )
