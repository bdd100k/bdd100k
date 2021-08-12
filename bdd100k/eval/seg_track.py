"""BDD100K tracking evaluation with CLEAR MOT metrics."""
import os
from typing import Callable, List

import motmetrics as mm
import numpy as np
from PIL import Image

from ..common.bitmask import (
    bitmask_intersection_rate,
    gen_blank_bitmask,
    parse_bitmask,
)
from ..common.utils import reorder_preds

Files = List[str]
FilesList = List[Files]
FilesFunc = Callable[
    [Files, Files, float, float, bool], List[mm.MOTAccumulator]
]


def acc_single_video_mots(  # pylint: disable=unused-argument
    gts: List[str],
    results: List[str],
    classes: List[str],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,
) -> List[mm.MOTAccumulator]:
    """Accumulate results for one video."""
    results = reorder_preds(gts, results)
    num_classes = len(classes)

    accs = [mm.MOTAccumulator(auto_id=True) for _ in range(num_classes)]
    for gt, result in zip(gts, results):
        assert os.path.isfile(gt)
        assert os.path.isfile(result)

        gt_bitmask = np.asarray(Image.open(gt), np.uint8)
        if not result:
            res_bitmask = gen_blank_bitmask(gt_bitmask.shape)
        else:
            res_bitmask = np.asarray(Image.open(result), dtype=np.uint8)
        gt_masks, gt_ids, gt_attrs, gt_cats = parse_bitmask(gt_bitmask)
        pred_masks, pred_ids, pred_attrs, pred_cats = parse_bitmask(
            res_bitmask
        )
        ious, iofs = bitmask_intersection_rate(gt_masks, pred_masks)

        gt_valids = np.logical_not(np.bitwise_and(gt_attrs, 3).astype(bool))
        pred_valids = np.logical_not(
            np.bitwise_and(pred_attrs, 3).astype(bool)
        )
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
                fps = np.ones(pred_ids_c.shape[0], dtype=bool)
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
