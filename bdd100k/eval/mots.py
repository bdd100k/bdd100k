"""BDD100K tracking evaluation with CLEAR MOT metrics."""
import os
from typing import List, Tuple

import motmetrics as mm
import numpy as np
from PIL import Image

from .mot import CLASSES

MAX_DET = 100


def parse_bitmasks(
    bitmask: np.ndarray,
) -> List[np.ndarray]:
    """Parse information from bitmasks and compress its value range.

    The compression works like: [4, 2, 9] --> [2, 1, 3]
    """
    category_map = bitmask[:, :, 0]
    attributes_map = bitmask[:, :, 1]
    instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

    # 0 is for the background
    instance_ids = np.sort(np.unique(instance_map[instance_map >= 1]))
    category_ids = np.zeros(instance_ids.shape, dtype=instance_ids.dtype)
    attributes = np.zeros(instance_ids.shape, dtype=instance_ids.dtype)

    masks = np.zeros(bitmask.shape[:2], dtype=np.int32)
    for i, instance_id in enumerate(instance_ids):
        mask_inds_i = instance_map == instance_id
        # 0 is for the background
        masks[mask_inds_i] = i + 1

        attributes_i = np.unique(attributes_map[mask_inds_i])
        assert attributes_i.shape[0] == 1
        attributes[i] = attributes_i[0]

        category_ids_i = np.unique(category_map[mask_inds_i])
        assert category_ids_i.shape[0] == 1
        category_ids[i] = category_ids_i[0]

    return [masks, instance_ids, attributes, category_ids]


def mask_intersection_rate(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the intersection over the area of the predicted box."""
    assert gt_masks.shape == pred_masks.shape
    m: int = np.max(gt_masks)
    n: int = min(np.max(pred_masks), MAX_DET)

    gt_masks = gt_masks.reshape(-1)
    pred_masks = pred_masks.reshape(-1)
    pred_masks[pred_masks > MAX_DET] = 0

    confusion = gt_masks * (1 + n) + pred_masks
    bin_num = (1 + m) * (1 + n)
    histogram = np.histogram(confusion, bins=bin_num, range=(0, bin_num))[0]
    conf_matrix = histogram.reshape(1 + m, 1 + n)
    gt_areas = conf_matrix.sum(axis=1)[1:]
    pred_areas = conf_matrix.sum(axis=0)[1:]
    conf_matrix = conf_matrix[1:, 1:]

    ious = np.zeros((m, n))
    iofs = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            inter = conf_matrix[i, j]
            union = gt_areas[i] + pred_areas[j] - conf_matrix[i, j]
            if union > 0:
                ious[i, j] = inter / union
            else:
                ious[i, j] = 0
            if pred_areas[j] > 0:
                iofs[i, j] = inter / pred_areas[j]
            else:
                iofs[i, j] = 0

    return ious, iofs


def acc_single_video_mots(
    gts: List[str],
    results: List[str],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
) -> List[mm.MOTAccumulator]:
    """Accumulate results for one video."""
    assert len(gts) == len(results)
    num_classes = len(CLASSES)

    accs = [mm.MOTAccumulator(auto_id=True) for _ in range(num_classes)]
    for gt, result in zip(gts, results):
        assert os.path.isfile(gt)
        assert os.path.isfile(result)

        gt_masks = np.asarray(Image.open(gt)).astype(np.int32)
        res_masks = np.asarray(Image.open(result)).astype(np.int32)
        gt_masks, gt_ids, gt_attrs, gt_cats = parse_bitmasks(gt_masks)
        pred_masks, pred_ids, pred_attrs, pred_cats = parse_bitmasks(res_masks)
        ious, iofs = mask_intersection_rate(gt_masks, pred_masks)

        gt_valids = np.logical_not((gt_attrs & 3).astype(np.bool8))
        pred_valids = np.logical_not((pred_attrs & 3).astype(np.bool8))
        for i in range(num_classes):
            # cats starts from 1 and i starts from 0
            gt_inds = (gt_cats == i + 1) * gt_valids
            pred_inds = (pred_cats == i + 1) * pred_valids
            gt_ids_c, pred_ids_c = gt_ids[gt_inds], pred_ids[pred_inds]

            if gt_ids_c.shape[0] == 0 and pred_ids_c.shape[0] != 0:
                distances = np.full((0, pred_ids_c.shape[0]), np.nan)
            elif gt_ids_c.shape[0] != 0 and pred_ids_c.shape[0] == 0:
                distances = np.full((gt_ids_c.shape[0], 0), np.nan)
            else:
                ious_c = ious[gt_inds, :][:, pred_inds]
                distances = 1 - ious_c
                distances = np.where(
                    distances > 1 - iou_thr, np.nan, distances
                )

            gt_invalid = np.logical_not(gt_valids)
            if (gt_invalid).any():
                # 1. assign gt and preds
                fps = np.ones(pred_ids_c.shape[0]).astype(np.bool8)
                le, ri = mm.lap.linear_sum_assignment(distances)
                for m, n in zip(le, ri):
                    if np.isfinite(distances[m, n]):
                        fps[n] = False
                # 2. ignore by iof
                iofs_c = iofs[gt_invalid, :][:, pred_inds]
                ignores = (iofs_c > ignore_iof_thr).any(axis=0)
                # 3. filter preds
                valid_inds = ~(fps & ignores)
                pred_ids_c = pred_ids_c[valid_inds]
                distances = distances[:, valid_inds]
            if distances.shape != (0, 0):
                accs[i].update(gt_ids_c, pred_ids_c, distances)
    return accs
