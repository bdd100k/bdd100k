"""Evaluation code for BDD100K lane marking.

************************************
Byte structure for lane marking:
+---+---+---+---+---+---+---+---+
| - | - | d | s | b | c | c | c |
+---+---+---+---+---+---+---+---+

d: direction
s: style
b: background
c: category

More details: bdd100k.label.label.py
************************************


Code adapted from:
https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/f_boundary.py

Source License

BSD 3-Clause License

Copyright (c) 2017,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.s
############################################################################

Based on:
----------------------------------------------------------------------------
A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
Copyright (c) 2016 Federico Perazzi
Licensed under the BSD License [see LICENSE for details]
Written by Federico Perazzi
----------------------------------------------------------------------------
"""
from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64, NDArrayU8
from scalabel.eval.result import AVERAGE, Result, Scores, ScoresList
from skimage.morphology import binary_dilation, disk  # type: ignore
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import reorder_preds
from ..label.label import lane_categories, lane_directions, lane_styles

BOUND_PIXELS = [1, 2, 5]


def eval_lane_per_threshold(
    gt_mask: NDArrayU8, pd_mask: NDArrayU8, bound_pix: int
) -> float:
    """Compute mean,recall and decay from per-threshold evaluation."""
    gt_dil = binary_dilation(gt_mask, disk(bound_pix))
    pd_dil = binary_dilation(pd_mask, disk(bound_pix))

    # Get the intersection
    gt_match = gt_mask * pd_dil
    pd_match = pd_mask * gt_dil

    # Area of the intersection
    n_gt = np.sum(gt_mask)
    n_pd = np.sum(pd_mask)

    # Compute precision and recall
    if n_pd == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_pd > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_pd == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(pd_match) / float(n_pd)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2.0 * precision * recall / (precision + recall)

    return f_score


def get_lane_class(
    byte: NDArrayU8, value: int, offset: int, width: int
) -> NDArrayU8:
    """Extract the lane class given offset, width and value."""
    assert byte.dtype is np.dtype(np.uint8)
    assert 0 <= value < (1 << 8)
    assert 0 <= offset < 8
    assert 0 < width <= 8
    lane_cls = np.equal(
        np.bitwise_and(
            np.right_shift(byte, offset), np.left_shift(1, width) - 1
        ),
        value,
    )
    return lane_cls  # type: ignore


def lane_class_func(
    offset: int, width: int
) -> Callable[[NDArrayU8, int], NDArrayU8]:
    """Get the function for extracting the specific lane class."""
    return partial(get_lane_class, offset=offset, width=width)


get_foreground = partial(get_lane_class, value=0, offset=3, width=1)
sub_task_funcs = dict(
    direction=lane_class_func(5, 1),
    style=lane_class_func(4, 1),
    category=lane_class_func(0, 3),
)
sub_task_cats: Dict[str, List[str]] = dict(
    direction=[label.name for label in lane_directions],
    style=[label.name for label in lane_styles],
    category=[label.name for label in lane_categories],
)


class LaneResult(Result):
    """The class for lane marking evaluation results."""

    F1_pix1: List[Dict[str, float]]
    F1_pix2: List[Dict[str, float]]
    F1_pix5: List[Dict[str, float]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "LaneResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the lane_mark data into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            for category, score in scores_list[-2].items():
                summary_dict[f"{metric}/{category}"] = score
            summary_dict[metric] = scores_list[-1][AVERAGE]
        return summary_dict


def eval_lane_per_frame(gt_path: str, pred_path: str) -> Dict[str, NDArrayF64]:
    """Compute mean,recall and decay from per-frame evaluation."""
    task2arr: Dict[str, NDArrayF64] = {}  # str -> 2d array
    gt_byte: NDArrayU8 = np.asarray(Image.open(gt_path), dtype=np.uint8)
    if not pred_path:
        pred_byte: NDArrayU8 = np.zeros_like(gt_byte, dtype=np.uint8)
    else:
        pred_byte = np.asarray(Image.open(pred_path), dtype=np.uint8)
    gt_foreground = get_foreground(gt_byte)
    pd_foreground = get_foreground(pred_byte)

    for task_name, class_func in sub_task_funcs.items():
        task_scores: List[List[float]] = []
        for value in range(len(sub_task_cats[task_name])):
            gt_mask = np.logical_and(class_func(gt_byte, value), gt_foreground)
            pd_mask = np.logical_and(
                class_func(pred_byte, value), pd_foreground
            )
            cat_scores = [
                eval_lane_per_threshold(gt_mask, pd_mask, bound_pixel)
                for bound_pixel in BOUND_PIXELS
            ]
            task_scores.append(cat_scores)
        task2arr[task_name] = np.array(task_scores)

    return task2arr


def merge_results(
    task2arr_list: List[Dict[str, NDArrayF64]]
) -> Dict[str, NDArrayF64]:
    """Merge F-score results from all images."""
    task2arr: Dict[str, NDArrayF64] = {
        task_name: np.stack(
            [task2arr_img[task_name] for task2arr_img in task2arr_list]
        ).mean(axis=0)
        for task_name in sub_task_cats
    }

    for task_name, arr2d in task2arr.items():
        arr2d *= 100
        arr_mean = arr2d.mean(axis=0, keepdims=True)
        task2arr[task_name] = np.concatenate([arr2d, arr_mean], axis=0)

    avg_arr: NDArrayF64 = np.stack([arr2d[-1] for arr2d in task2arr.values()])
    task2arr[AVERAGE] = avg_arr.mean(axis=0)

    return task2arr


def generate_results(task2arr: Dict[str, NDArrayF64]) -> LaneResult:
    """Render the evaluation results."""
    res_dict: Dict[str, ScoresList] = {
        f"F1_pix{bound_pixel}": [{} for _ in range(5)]
        for bound_pixel in BOUND_PIXELS
    }

    cur_ind = 0
    for task_name, arr2d in task2arr.items():
        if task_name == AVERAGE:
            continue
        for cat_name, arr1d in zip(sub_task_cats[task_name], arr2d):
            for bound_pixel, f_score in zip(BOUND_PIXELS, arr1d):
                res_dict[f"F1_pix{bound_pixel}"][cur_ind][cat_name] = f_score
        cur_ind += 1

    for task_name, arr2d in task2arr.items():
        task_name = task_name.upper()
        if task_name == AVERAGE:
            continue
        arr1d = arr2d[-1]
        for bound_pixel, f_score in zip(BOUND_PIXELS, arr1d):
            res_dict[f"F1_pix{bound_pixel}"][-2][task_name] = f_score

    for bound_pixel, f_score in zip(BOUND_PIXELS, task2arr[AVERAGE]):
        res_dict[f"F1_pix{bound_pixel}"][-1][AVERAGE] = f_score

    return LaneResult(**res_dict)


def evaluate_lane_marking(
    gt_paths: List[str],
    pred_paths: List[str],
    nproc: int = NPROC,
    with_logs: bool = True,
) -> LaneResult:
    """Evaluate F-score for lane marking from input folders."""
    if with_logs:
        logger.info("evaluating...")
    pred_paths = reorder_preds(gt_paths, pred_paths)
    if nproc > 1:
        with Pool(nproc) as pool:
            task2arr_list = pool.starmap(
                eval_lane_per_frame,
                tqdm(zip(gt_paths, pred_paths), total=len(gt_paths)),
            )
    else:
        task2arr_list = [
            eval_lane_per_frame(gt_path, pred_path)
            for gt_path, pred_path in tqdm(
                zip(gt_paths, pred_paths), total=len(gt_paths)
            )
        ]
    if with_logs:
        logger.info("accumulating...")
    task2arr = merge_results(task2arr_list)
    result = generate_results(task2arr)
    return result
