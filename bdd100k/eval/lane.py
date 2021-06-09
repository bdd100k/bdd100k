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
from typing import Callable, Dict, List

import numpy as np
from PIL import Image
from skimage.morphology import (  # type: ignore
    binary_dilation,
    disk,
    skeletonize,
)
from tabulate import tabulate
from tqdm import tqdm

from ..common.utils import reorder_preds
from ..label.label import lane_categories, lane_directions, lane_styles

AVG = "avg"
TOTAL = "total"


def eval_lane_per_threshold(
    gt_mask: np.ndarray, pd_mask: np.ndarray, bound_th: float = 0.008
) -> float:
    """Compute mean,recall and decay from per-threshold evaluation."""
    bound_pix = (
        bound_th
        if bound_th >= 1
        else np.ceil(bound_th * np.linalg.norm(gt_mask.shape))
    )

    gt_skeleton = skeletonize(gt_mask)
    pd_skeleton = skeletonize(pd_mask)

    gt_dil = binary_dilation(gt_skeleton, disk(bound_pix))
    pd_dil = binary_dilation(pd_skeleton, disk(bound_pix))

    # Get the intersection
    gt_match = gt_skeleton * pd_dil
    pd_match = pd_skeleton * gt_dil

    # Area of the intersection
    n_gt = np.sum(gt_skeleton)
    n_pd = np.sum(pd_skeleton)

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
    byte: np.ndarray, value: int, offset: int, width: int
) -> np.ndarray:
    """Extract the lane class given offset, width and value."""
    assert byte.dtype == "uint8"
    assert 0 <= value < (1 << 8)
    assert 0 <= offset < 8
    assert 0 < width <= 8
    lane_cls = (((byte >> offset) & ((1 << width) - 1)) == value).astype(bool)
    return lane_cls  # type: ignore


def lane_class_func(
    offset: int, width: int
) -> Callable[[np.ndarray, int], np.ndarray]:
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


def eval_lane_per_frame(
    gt_path: str, pred_path: str, bound_ths: List[float]
) -> Dict[str, np.ndarray]:
    """Compute mean,recall and decay from per-frame evaluation."""
    task2arr: Dict[str, np.ndarray] = dict()  # str -> 2d array
    gt_byte = np.asarray(Image.open(gt_path), dtype=np.uint8)
    if not pred_path:
        pred_byte = np.zeros_list(gt_byte, dtype=np.uint8)
    else:
        pred_byte = np.asarray(Image.open(pred_path), dtype=np.uint8)
    gt_foreground = get_foreground(gt_byte)
    pd_foreground = get_foreground(pred_byte)

    for task_name, class_func in sub_task_funcs.items():
        task_scores: List[List[float]] = []
        for value in range(len(sub_task_cats[task_name])):
            gt_mask = class_func(gt_byte, value) & gt_foreground
            pd_mask = class_func(pred_byte, value) & pd_foreground
            cat_scores = [
                eval_lane_per_threshold(gt_mask, pd_mask, bound_th)
                for bound_th in bound_ths
            ]
            task_scores.append(cat_scores)
        task2arr[task_name] = np.array(task_scores)

    return task2arr


def merge_results(
    task2arr_list: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Merge F-score results from all images."""
    task2arr: Dict[str, np.ndarray] = {
        task_name: np.stack(
            [task2arr_img[task_name] for task2arr_img in task2arr_list]
        ).mean(axis=0)
        for task_name in sub_task_cats
    }

    for task_name, arr2d in task2arr.items():
        arr2d *= 100
        arr_mean = arr2d.mean(axis=0, keepdims=True)
        task2arr[task_name] = np.concatenate([arr_mean, arr2d], axis=0)

    avg_arr = np.stack([arr2d[-1] for arr2d in task2arr.values()])
    task2arr[TOTAL] = avg_arr.mean(axis=0, keepdims=True)

    return task2arr


def create_table(
    task2arr: Dict[str, np.ndarray],
    all_task_cats: Dict[str, List[str]],
    bound_ths: List[float],
) -> None:
    """Render the evaluation results."""
    table = []
    headers = ["task", "class"] + [str(th) for th in bound_ths]
    for task_name in sorted(sub_task_cats.keys()) + [TOTAL]:
        arr2d = task2arr[task_name]
        task_list, cat_list, num_strs = [], [], []
        for i, cat_name in enumerate(all_task_cats[task_name]):
            task_name_temp = task_name if i == arr2d.shape[0] // 2 else " "
            task_list.append("{}".format(task_name_temp))
            cat_list.append(cat_name.replace(" ", "_"))
        task_str = "\n".join(task_list)
        cat_str = "\n".join(cat_list)

        for j in range(len(bound_ths)):
            num_list = []
            for i in range(len(all_task_cats[task_name])):
                num_list.append("{:.1f}".format(arr2d[i, j]))
            num_str = "\n".join(num_list)
            num_strs.append(num_str)

        table.append([task_str, cat_str] + num_strs)

    print(tabulate(table, headers, tablefmt="grid", stralign="center"))


def render_results(
    task2arr: Dict[str, np.ndarray],
    all_task_cats: Dict[str, List[str]],
    bound_ths: List[float],
) -> Dict[str, float]:
    """Render the evaluation results."""
    f_score_dict: Dict[str, float] = dict()
    for task_name, arr2d in task2arr.items():
        for cat_name, arr1d in zip(all_task_cats[task_name], arr2d):
            for bound_th, f_score in zip(bound_ths, arr1d):
                f_score_dict[
                    "{:.1f}_{}_{}".format(
                        bound_th, task_name, cat_name.replace(" ", "_")
                    )
                ] = f_score
    f_score_dict["average"] = task2arr[TOTAL].mean()
    return f_score_dict


def evaluate_lane_marking(
    gt_paths: List[str],
    pred_paths: List[str],
    bound_ths: List[float],
    nproc: int = 4,
) -> Dict[str, float]:
    """Evaluate F-score for lane marking from input folders."""
    pred_paths = reorder_preds(gt_paths, pred_paths)
    with Pool(nproc) as pool:
        task2arr_list = pool.starmap(
            partial(eval_lane_per_frame, bound_ths=bound_ths),
            tqdm(zip(gt_paths, pred_paths), total=len(gt_paths)),
        )
    task2arr = merge_results(task2arr_list)

    all_task_cats = sub_task_cats.copy()
    for cats in all_task_cats.values():
        cats.append(AVG)
    all_task_cats.update({TOTAL: [AVG]})

    create_table(task2arr, all_task_cats, bound_ths)
    f_score_dict = render_results(task2arr, all_task_cats, bound_ths)
    return f_score_dict
