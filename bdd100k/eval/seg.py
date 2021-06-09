"""Evaluation procedures for semantic segmentation.

For dataset with `n` classes, we treat the index `n` as the ignored class.
When compute IoUs, this ignored class is considered.
However, IoU(ignored) doesn't influence mIoU.
"""

from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import reorder_preds
from ..label.label import drivables, labels
from ..label.to_mask import IGNORE_LABEL


def fast_hist(
    groundtruth: np.ndarray, prediction: np.ndarray, size: int
) -> np.ndarray:
    """Compute the histogram."""
    prediction = prediction.copy()
    prediction[prediction >= size] = size - 1

    k = (groundtruth >= 0) & (groundtruth < size)
    return np.bincount(  # type: ignore
        size * groundtruth[k].astype(int) + prediction[k], minlength=size ** 2
    ).reshape(size, size)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    """Calculate per class iou."""
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    # Last class as `ignored`
    return ious[:-1]  # type: ignore


def per_image_hist(
    gt_path: str, pred_path: str = "", num_classes: int = 2
) -> Tuple[np.ndarray, Set[int]]:
    """Calculate per image hist."""
    assert num_classes >= 2
    assert num_classes <= IGNORE_LABEL
    gt = np.asarray(Image.open(gt_path, "r"), dtype=np.uint8)
    gt = gt.copy()
    gt[gt == IGNORE_LABEL] = num_classes - 1
    gt_id_set = set(np.unique(gt).tolist())

    if not pred_path:
        pred = np.ones_like(gt, dtype=np.uint8) * (num_classes - 1)
    else:
        pred = np.asarray(Image.open(pred_path, "r"), dtype=np.uint8)
    hist = fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_segmentation(
    gt_paths: List[str],
    pred_paths: List[str],
    mode: str = "sem_seg",
    nproc: int = 4,
) -> Dict[str, float]:
    """Evaluate segmentation IoU from input folders."""
    assert mode in ["sem_seg", "drivable"]
    logger.info("Found %d results", len(gt_paths))
    label_defs = {
        "sem_seg": labels,
        "drivable": drivables,
    }[mode]
    categories = [label.name for label in label_defs if label.trainId != 255]
    num_classes = {
        "sem_seg": len(labels) + 1,  # add an ignored class
        "drivable": len(drivables),  # `background` as `ignored`
    }[mode]

    sorted_results = reorder_preds(gt_paths, pred_paths)

    with Pool(nproc) as pool:
        hist_and_gt_id_sets = pool.starmap(
            partial(per_image_hist, num_classes=num_classes),
            tqdm(zip(gt_paths, sorted_results), total=len(gt_paths)),
        )
    hist = np.zeros((num_classes, num_classes))
    gt_id_set = set()
    for (hist_, gt_id_set_) in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])

    iou_dict = dict(miou=miou)
    logger.info("mIoU: {:.2f}".format(miou))
    for category, iou in zip(categories, ious):
        iou_dict[category] = iou
        logger.info("{}: {:.2f}".format(category, iou))
    return iou_dict


def evaluate_drivable(
    gt_paths: List[str], pred_paths: List[str], nproc: int = 4
) -> Dict[str, float]:
    """Evaluate drivable area."""
    return evaluate_segmentation(
        gt_paths, pred_paths, mode="drivable", nproc=nproc
    )


def evaluate_sem_seg(
    gt_paths: List[str], pred_paths: List[str], nproc: int = 4
) -> Dict[str, float]:
    """Evaluate semantic segmentation."""
    return evaluate_segmentation(
        gt_paths, pred_paths, mode="sem_seg", nproc=nproc
    )
