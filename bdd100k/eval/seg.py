"""Evaluation procedures for semantic segmentation."""

import os.path as osp
from typing import Dict

import numpy as np
import toml
from PIL import Image

from ..common.logger import logger
from ..common.utils import DEFAULT_COCO_CONFIG, DEFAULT_SEG_STRING, list_files


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


def evaluate_segmentation(
    gt_dir: str,
    res_dir: str,
    cfg_file: str = DEFAULT_COCO_CONFIG,
    cfg_str: str = DEFAULT_SEG_STRING,
    mode: str = "sem_seg",
) -> Dict[str, float]:
    """Evaluate segmentation IoU from input folders."""
    assert mode in ["sem_seg", "drivable", "lane_mark"]
    categories = toml.load(cfg_file)[cfg_str][mode]
    num_classes = len(categories)

    gt_imgs = list_files(gt_dir, ".png")
    res_imgs = list_files(res_dir, ".png")
    logger.info("Found %d results", len(gt_imgs))
    for gt_img, res_img in zip(gt_imgs, res_imgs):
        assert gt_img == res_img

    hist = np.zeros((num_classes, num_classes))
    gt_id_set = set()
    for i, img in enumerate(gt_imgs):
        gt_path = osp.join(gt_dir, img)
        res_path = osp.join(res_dir, img)
        gt = np.asarray(Image.open(gt_path, "r"))[..., 0]
        gt_id_set.update(np.unique(gt).tolist())
        pred = np.asanyarray(Image.open(res_path, "r"))[..., 0]
        hist += fast_hist(gt.flatten(), pred.flatten(), num_classes)
        if (i + 1) % 100 == 0:
            logger.info("Finished %d", (i + 1))
    if 255 in gt_id_set:
        gt_id_set.remove(255)
    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])

    iou_dict = dict(miou=miou)
    logger.info("{:.2f}".format(miou))
    for category, iou in zip(categories, ious):
        iou_dict[category] = iou
        logger.info("{}: {:.2f}".format(category, iou))
    return iou_dict


def evaluate_drivable(gt_dir: str, result_dir: str) -> Dict[str, float]:
    """Evaluate drivable area."""
    return evaluate_segmentation(gt_dir, result_dir, mode="drivable")


def evaluate_lane_marking(gt_dir: str, result_dir: str) -> Dict[str, float]:
    """Evaluate drivable area."""
    return evaluate_segmentation(gt_dir, result_dir, mode="lane_mark")
