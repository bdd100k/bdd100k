"""Evaluation procedures for semantic segmentation.

For dataset with `n` classes, we treat the index `n` as the ignored class.
When compute IoUs, this ignored class is considered.
However, IoU(ignored) doesn't influence mIoU.
"""

import os.path as osp
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common.logger import logger
from ..label.label import drivables, labels
from ..label.to_mask import IGNORE_LABEL


def fast_hist(
    groundtruth: np.ndarray, prediction: np.ndarray, size: int
) -> np.ndarray:
    """Compute the histogram."""
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
    gt_path: str, res_path: str = "", num_classes: int = 1
) -> Tuple[np.ndarray, Set[int]]:
    """Calculate per image hist."""
    gt = np.asarray(Image.open(gt_path, "r"), dtype=np.uint8)
    gt = gt.copy()
    gt[gt == IGNORE_LABEL] = num_classes - 1
    gt_id_set = set(np.unique(gt).tolist())

    if not res_path:
        pred = np.ones(gt.shape, dtype=np.uint8) * (num_classes - 1)
    else:
        pred = np.asarray(Image.open(res_path, "r"), dtype=np.uint8)
    pred = pred.copy()
    pred[pred >= num_classes] = num_classes - 1
    hist = fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_segmentation(
    gts: List[str],
    results: List[str],
    mode: str = "sem_seg",
    nproc: int = 4,
) -> Dict[str, float]:
    """Evaluate segmentation IoU from input folders."""
    assert mode in ["sem_seg", "drivable"]
    logger.info("Found %d results", len(gts))
    label_defs = {
        "sem_seg": labels,
        "drivable": drivables,
    }[mode]
    categories = [label.name for label in label_defs if label.trainId != 255]
    num_classes = {
        "sem_seg": len(labels) + 1,  # add an ignored class
        "drivable": len(drivables),  # `background` as `ignored`
    }[mode]

    res_map: Dict[str, str] = {
        osp.splitext(osp.split(res_path)[-1])[0]: res_path
        for res_path in results
    }
    sorted_results: List[str] = []
    for gt_path in gts:
        gt_name = osp.splitext(osp.split(gt_path)[-1])[0]
        if gt_name in res_map:
            sorted_results.append(res_map[gt_name])
        else:
            sorted_results.append("")

    with Pool(nproc) as pool:
        hist_and_gt_id_sets = pool.starmap(
            partial(per_image_hist, num_classes=num_classes),
            tqdm(zip(gts, sorted_results), total=len(gts)),
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
    gts: List[str], results: List[str], nproc: int = 4
) -> Dict[str, float]:
    """Evaluate drivable area."""
    return evaluate_segmentation(gts, results, mode="drivable", nproc=nproc)


def evaluate_sem_seg(
    gts: List[str], results: List[str], nproc: int = 4
) -> Dict[str, float]:
    """Evaluate semantic segmentation."""
    return evaluate_segmentation(gts, results, mode="sem_seg", nproc=nproc)
