"""Evaluation procedures for semantic segmentation."""

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..common.logger import logger


def fast_hist(
    groundtruth: np.ndarray, prediction: np.ndarray, size: int
) -> np.ndarray:
    """Compute the histogram."""
    k = (groundtruth >= 0) & (groundtruth < size)
    return np.bincount(
        size * groundtruth[k].astype(int) + prediction[k], minlength=size ** 2
    ).reshape(size, size)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    """Calculate per class iou."""
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_png(folder: str) -> List[str]:
    """List png files."""
    paths = []
    for root, _, files in os.walk(folder, topdown=True):
        paths.extend(
            [osp.join(root, f) for f in files if osp.splitext(f)[1] == ".png"]
        )
    return paths


def evaluate_segmentation(
    gt_dir: str, result_dir: str, num_classes: int, key_length: int
) -> Tuple[np.ndarray, float]:
    """Evaluate segmentation IoU from input folders."""
    gt_dict = {osp.split(p)[1][:key_length]: p for p in find_all_png(gt_dir)}
    result_dict = {
        osp.split(p)[1][:key_length]: p for p in find_all_png(result_dir)
    }
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError(
            "Result folder only has {} of {} ground truth files.".format(
                len(result_gt_keys), len(gt_dict)
            )
        )
    logger.info("Found %d results", len(result_dict))
    logger.info("Evaluating %d results", len(gt_dict))
    hist = np.zeros((num_classes, num_classes))
    gt_id_set = set()
    for i, key in enumerate(sorted(gt_dict.keys())):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, "r"))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, "r"))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        if (i + 1) % 100 == 0:
            logger.info("Finished %d %f", (i + 1), per_class_iu(hist) * 100)
    if 255 in gt_id_set:
        gt_id_set.remove(255)
    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])

    logger.info("{:.2f}".format(miou))
    logger.info(", ".join(["{:.2f}".format(n) for n in list(ious)]))
    return ious, miou


def evaluate_drivable(gt_dir: str, result_dir: str) -> None:
    """Evaluate drivable area."""
    evaluate_segmentation(gt_dir, result_dir, 3, 17)
