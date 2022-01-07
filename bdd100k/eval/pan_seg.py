"""Evaluation code for BDD100K panoptic segmentation.

############################################################################
Code adapted from:
https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py

Copyright (c) 2018, Alexander Kirillov
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
############################################################################
"""
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import AbstractSet, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayU8
from scalabel.eval.result import OVERALL, Result, Scores, ScoresList
from scalabel.label.coco_typing import PanopticCatType
from tqdm import tqdm

from ..common.bitmask import (
    bitmask_intersection_rate,
    gen_blank_bitmask,
    parse_bitmask,
)
from ..common.logger import logger
from ..common.utils import reorder_preds
from ..label.label import labels

STUFF = "STUFF"
THING = "THING"


class PanSegResult(Result):
    """The class for panoptic segmentation evaluation results."""

    PQ: List[Dict[str, float]]
    SQ: List[Dict[str, float]]
    RQ: List[Dict[str, float]]
    N: List[Dict[str, int]]  # pylint: disable=invalid-name

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "PanSegResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the pan_seg data into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            summary_dict[f"{metric}/{STUFF}"] = scores_list[1][STUFF]
            summary_dict[f"{metric}/{THING}"] = scores_list[1][THING]
            summary_dict[metric] = scores_list[-1][OVERALL]
        return summary_dict


class PQStatCat:
    """PQ statistics for each category."""

    def __init__(self) -> None:
        """Initialize method."""
        self.iou: float = 0.0
        self.tp: int = 0  # pylint: disable=invalid-name
        self.fp: int = 0  # pylint: disable=invalid-name
        self.fn: int = 0  # pylint: disable=invalid-name

    def __iadd__(self, pq_stat_cat: "PQStatCat") -> "PQStatCat":
        """Adding definition."""
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat:
    """PQ statistics for an image of the whole dataset."""

    def __init__(self) -> None:
        """Initialize the PQStatCat dict."""
        self.pq_per_cats: Dict[int, PQStatCat] = defaultdict(PQStatCat)

    def __getitem__(self, category_id: int) -> PQStatCat:
        """Get a PQStatCat object given category."""
        return self.pq_per_cats[category_id]

    def __iadd__(self, pq_stat: "PQStat") -> "PQStat":
        """Adding definition."""
        for category_id, pq_stat_cat in pq_stat.pq_per_cats.items():
            self.pq_per_cats[category_id] += pq_stat_cat
        return self

    def pq_average(
        self, categories: List[PanopticCatType]
    ) -> Dict[str, float]:
        """Calculate averatge metrics over categories."""
        pq, sq, rq, n = 0.0, 0.0, 0.0, 0
        for category in categories:
            category_id = category["id"]
            iou = self.pq_per_cats[category_id].iou
            tp = self.pq_per_cats[category_id].tp
            fp = self.pq_per_cats[category_id].fp
            fn = self.pq_per_cats[category_id].fn

            if tp + fp + fn == 0:
                continue
            pq += (iou / (tp + 0.5 * fp + 0.5 * fn)) * 100
            sq += (iou / tp if tp != 0 else 0) * 100
            rq += (tp / (tp + 0.5 * fp + 0.5 * fn)) * 100
            n += 1

        if n > 0:
            return dict(PQ=pq / n, SQ=sq / n, RQ=rq / n, N=n)
        return dict(PQ=0, SQ=0, RQ=0, N=0)


def pq_per_image(gt_path: str, pred_path: str = "") -> PQStat:
    """Calculate PQStar for each image."""
    gt_bitmask: NDArrayU8 = np.asarray(Image.open(gt_path), dtype=np.uint8)
    if not pred_path:
        pred_bitmask = gen_blank_bitmask(gt_bitmask.shape)
    else:
        pred_bitmask = np.asarray(Image.open(pred_path), dtype=np.uint8)

    gt_masks, gt_ids, gt_attrs, gt_cats = parse_bitmask(gt_bitmask)
    pred_masks, pred_ids, pred_attrs, pred_cats = parse_bitmask(pred_bitmask)

    gt_valids = np.logical_not(np.bitwise_and(gt_attrs, 3).astype(bool))
    pred_valids = np.logical_not(np.bitwise_and(pred_attrs, 3).astype(bool))

    ious, iofs = bitmask_intersection_rate(gt_masks, pred_masks)
    cat_equals = gt_cats.reshape(-1, 1) == pred_cats.reshape(1, -1)
    ious *= cat_equals

    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)
    inv_iofs = 1 - iofs[gt_valids].sum(axis=0)

    pq_stat = PQStat()
    pred_matched = set()
    for i in range(len(gt_ids)):
        if not gt_valids[i]:
            continue
        cat_i = gt_cats[i]
        if max_ious[i] <= 0.5 or not pred_valids[max_idxs[i]]:
            pq_stat[cat_i].fn += 1
        else:
            pq_stat[cat_i].tp += 1
            pq_stat[cat_i].iou += max_ious[i]
            pred_matched.add(max_idxs[i])

    for j in range(len(pred_ids)):
        if not pred_valids[j] or j in pred_matched or inv_iofs[j] > 0.5:
            continue
        pq_stat[pred_cats[j]].fp += 1
    return pq_stat


def evaluate_pan_seg(
    gt_paths: List[str],
    pred_paths: List[str],
    nproc: int = NPROC,
    with_logs: bool = True,
) -> PanSegResult:
    """Evaluate panoptic segmentation with BDD100K format."""
    start_time = time.time()
    if with_logs:
        logger.info("evaluating...")
    pred_paths = reorder_preds(gt_paths, pred_paths)
    if nproc > 1:
        with Pool(nproc) as pool:
            pq_stats = pool.starmap(
                pq_per_image,
                tqdm(zip(gt_paths, pred_paths), total=len(gt_paths)),
            )
    else:
        pq_stats = [
            pq_per_image(gt_path, pred_path)
            for gt_path, pred_path in tqdm(
                zip(gt_paths, pred_paths), total=len(gt_paths)
            )
        ]
    pq_stat = PQStat()
    for pq_stat_ in pq_stats:
        pq_stat += pq_stat_

    if with_logs:
        logger.info("accumulating...")
    categories: List[PanopticCatType] = [
        PanopticCatType(
            id=label.id,
            name=label.name,
            supercategory=label.category,
            isthing=label.hasInstances,
            color=label.color,
        )
        for label in labels
    ]
    categories = categories[1:]
    categories_stuff = [
        category for category in categories if not category["isthing"]
    ]
    categories_thing = [
        category for category in categories if category["isthing"]
    ]
    basic_category_names = [category["name"] for category in categories]

    res_dict: Dict[str, ScoresList] = {}
    for category_name, category in zip(basic_category_names, categories):
        result = pq_stat.pq_average([category])
        for metric, score in result.items():
            if metric not in res_dict:
                res_dict[metric] = [{}, {}, {}]
            res_dict[metric][0][category_name] = score

    result = pq_stat.pq_average(categories_stuff)
    for metric, score in result.items():
        res_dict[metric][1][STUFF] = score
    result = pq_stat.pq_average(categories_thing)
    for metric, score in result.items():
        res_dict[metric][1][THING] = score
    result = pq_stat.pq_average(categories)
    for metric, score in result.items():
        res_dict[metric][2][OVERALL] = score

    t_delta = time.time() - start_time
    if with_logs:
        logger.info("Time elapsed: %0.2f seconds", t_delta)

    return PanSegResult(**res_dict)
