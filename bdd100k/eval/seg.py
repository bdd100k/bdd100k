"""Evaluation procedures for semantic segmentation."""

import os.path as osp
from functools import partial
from multiprocessing import Pool
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import list_files
from ..label.label import drivables, labels


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


def per_image_hist(gt_path, res_path, num_classes):
    gt = np.asarray(Image.open(gt_path, "r"))
    gt_id_set = set(np.unique(gt).tolist())
    pred = np.asanyarray(Image.open(res_path, "r"))
    hist = fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_segmentation(
    gt_dir: str,
    res_dir: str,
    mode: str = "sem_seg",
    nproc: int = 4,
) -> Dict[str, float]:
    """Evaluate segmentation IoU from input folders."""
    assert mode in ["sem_seg", "drivable"]
    label_defs = {
        "sem_seg": labels,
        "drivable": drivables,
    }[mode]
    categories = [label.name for label in label_defs if label.trainId != 255]
    num_classes = len(categories)

    gt_imgs = list_files(gt_dir, ".png")
    res_imgs = list_files(res_dir, ".png")
    logger.info("Found %d results", len(gt_imgs))
    for gt_img, res_img in zip(gt_imgs, res_imgs):
        assert gt_img == res_img

    gt_paths = [osp.join(gt_dir, img) for img in gt_imgs]
    res_paths = [osp.join(res_dir, img) for img in gt_imgs]

    with Pool(nproc) as pool:
        hist_and_gt_id_sets = pool.starmap(
            partial(per_image_hist, num_classes=num_classes),
            tqdm(zip(gt_paths, res_paths), total=len(gt_imgs)),
        )
    hist = np.zeros((num_classes, num_classes))
    gt_id_set = set()
    for (hist_, gt_id_set_) in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    if 255 in gt_id_set:
        gt_id_set.remove(255)
    if mode == "drivable":
        background = len(categories) - 1
        if background in gt_id_set:
            gt_id_set.remove(background)
        categories.remove("background")
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
    gt_dir: str, result_dir: str, nproc: int = 4
) -> Dict[str, float]:
    """Evaluate drivable area."""
    return evaluate_segmentation(
        gt_dir, result_dir, mode="drivable", nproc=nproc
    )
