"""BDD100K tracking evaluation with CLEAR MOT metrics."""
import os
import time
from functools import partial
from multiprocessing import Pool
from typing import List

import motmetrics as mm
import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayU8
from scalabel.eval.mot import (
    METRIC_MAPS,
    TrackResult,
    aggregate_accs,
    evaluate_single_class,
    generate_results,
)
from scalabel.label.typing import Config
from scalabel.label.utils import get_leaf_categories, get_parent_categories

from ..common.bitmask import (
    bitmask_intersection_rate,
    gen_blank_bitmask,
    parse_bitmask,
)
from ..common.logger import logger
from ..common.utils import reorder_preds

Files = List[str]
FilesList = List[Files]


def acc_single_video_mots(
    gts: Files,
    results: Files,
    classes: List[str],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,  # pylint: disable=unused-argument
) -> List[mm.MOTAccumulator]:
    """Accumulate results for one video."""
    results = reorder_preds(gts, results)
    num_classes = len(classes)

    accs = [mm.MOTAccumulator(auto_id=True) for _ in range(num_classes)]
    for gt, result in zip(gts, results):
        assert os.path.isfile(gt)
        assert os.path.isfile(result)

        gt_bitmask: NDArrayU8 = np.asarray(Image.open(gt), np.uint8)
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
                fps: NDArrayU8 = np.ones(pred_ids_c.shape[0], dtype=bool)
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


def evaluate_seg_track(
    gts: FilesList,
    results: FilesList,
    config: Config,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    nproc: int = NPROC,
) -> TrackResult:
    """Evaluate CLEAR MOT metrics for MOTS.

    Args:
        gts: the ground truth annotation files.
        results: the prediction result files.
        config: Config object
        iou_thr: Minimum IoU for a bounding box to be considered a positive.
        ignore_iof_thr: Min. Intersection over foreground with ignore regions.
        nproc: processes number for loading files

    Returns:
        TrackResult: evaluation results.
    """
    logger.info("Tracking evaluation with CLEAR MOT metrics.")
    t = time.time()
    assert len(gts) == len(results)

    classes = get_leaf_categories(config.categories)
    super_classes = get_parent_categories(config.categories)

    logger.info("evaluating...")
    class_names = [c.name for c in classes]
    if nproc > 1:
        with Pool(nproc) as pool:
            video_accs = pool.starmap(
                partial(
                    acc_single_video_mots,
                    classes=class_names,
                    ignore_iof_thr=ignore_iof_thr,
                ),
                zip(gts, results),
            )
    else:
        video_accs = [
            acc_single_video_mots(
                gt, result, class_names, iou_thr, ignore_iof_thr
            )
            for gt, result in zip(gts, results)
        ]

    class_names, metric_names, class_accs = aggregate_accs(
        video_accs, classes, super_classes
    )

    logger.info("accumulating...")
    if nproc > 1:
        with Pool(nproc) as pool:
            flat_dicts = pool.starmap(
                evaluate_single_class, zip(metric_names, class_accs)
            )
    else:
        flat_dicts = [
            evaluate_single_class(names, accs)
            for names, accs in zip(metric_names, class_accs)
        ]

    metrics = list(METRIC_MAPS.values())
    result = generate_results(
        flat_dicts, class_names, metrics, classes, super_classes
    )
    t = time.time() - t
    logger.info("evaluation finishes with %.1f s.", t)
    return result
