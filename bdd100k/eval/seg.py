"""Evaluation procedures for semantic segmentation.

For dataset with `n` classes, we treat the index `n` as the ignored class.
When compute IoUs, this ignored class is considered.
However, IoU(ignored) doesn't influence mIoU.
"""

from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64, NDArrayI32, NDArrayU8
from scalabel.eval.result import AVERAGE, Result, Scores, ScoresList
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import reorder_preds
from ..label.label import drivables, labels
from ..label.to_mask import IGNORE_LABEL


class SegResult(Result):
    """The class for general segmentation evaluation results."""

    IoU: List[Dict[str, float]]
    Acc: List[Dict[str, float]]
    fIoU: float
    pAcc: float

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "SegResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the seg result into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            if not isinstance(scores_list, list):
                summary_dict[metric] = scores_list
            else:
                summary_dict["m" + metric] = scores_list[-1][AVERAGE]
        return summary_dict


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
    return np.bincount(
        size * groundtruth[k].astype(int) + prediction[k], minlength=size**2
    ).reshape(size, size)


def per_class_iou(hist: NDArrayI32) -> NDArrayF64:
    """Calculate per class iou."""
    ious: NDArrayF64 = np.diag(hist) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist)
    )
    ious[np.isnan(ious)] = 0
    # Last class as `ignored`
    return ious[:-1]  # type: ignore


def per_class_acc(hist: NDArrayI32) -> NDArrayF64:
    """Calculate per class accuracy."""
    accs: NDArrayF64 = np.diag(hist) / hist.sum(axis=0)
    accs[np.isnan(accs)] = 0
    # Last class as `ignored`
    return accs[:-1]  # type: ignore


def whole_acc(hist: NDArrayI32) -> float:
    """Calculate whole accuray."""
    hist = hist[:-1]
    return cast(float, np.diag(hist).sum() / hist.sum())


def freq_iou(hist: NDArrayI32) -> float:
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
    gt: NDArrayU8 = np.asarray(Image.open(gt_path), dtype=np.uint8)
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
    with_logs: bool = True,
) -> SegResult:
    """Evaluate segmentation IoU from input folders."""
    assert mode in ["sem_seg", "drivable"]
    if with_logs:
        logger.info("Found %d results", len(gt_paths))
    label_defs = {
        "sem_seg": labels,
        "drivable": drivables,
    }[mode]
    label_sort = sorted(label_defs, key=lambda label: int(label.trainId))
    categories = [label.name for label in label_sort if label.trainId != 255]
    num_classes = {
        "sem_seg": len(categories) + 1,  # add an `ignored` class
        "drivable": len(drivables),  # `background` as `ignored`
    }[mode]

    if with_logs:
        logger.info("evaluating...")
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

    if with_logs:
        logger.info("accumulating...")
    hist: NDArrayI32 = np.zeros((num_classes, num_classes), dtype=np.int32)
    gt_id_set = set()
    for (hist_, gt_id_set_) in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    ious = per_class_iou(hist)
    accs = per_class_acc(hist)
    IoUs = [  # pylint: disable=invalid-name
        {cat_name: 100 * score for cat_name, score in zip(categories, ious)},
        {AVERAGE: np.multiply(np.mean(ious[list(gt_id_set)]), 100)},
    ]
    Accs = [  # pylint: disable=invalid-name
        {cat_name: 100 * score for cat_name, score in zip(categories, accs)},
        {AVERAGE: np.multiply(np.mean(accs[list(gt_id_set)]), 100)},
    ]
    res_dict: Dict[str, Union[float, ScoresList]] = dict(
        IoU=IoUs,
        Acc=Accs,
        fIoU=np.multiply(freq_iou(hist), 100),  # pylint: disable=invalid-name
        pAcc=np.multiply(whole_acc(hist), 100),  # pylint: disable=invalid-name
    )

    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    return SegResult(**res_dict)


def evaluate_drivable(
    gt_paths: List[str],
    pred_paths: List[str],
    nproc: int = NPROC,
    with_logs: bool = True,
) -> SegResult:
    """Evaluate drivable area."""
    return evaluate_segmentation(
        gt_paths, pred_paths, mode="drivable", nproc=nproc, with_logs=with_logs
    )


def evaluate_sem_seg(
    gt_paths: List[str],
    pred_paths: List[str],
    nproc: int = NPROC,
    with_logs: bool = True,
) -> SegResult:
    """Evaluate semantic segmentation."""
    return evaluate_segmentation(
        gt_paths, pred_paths, mode="sem_seg", nproc=nproc, with_logs=with_logs
    )
