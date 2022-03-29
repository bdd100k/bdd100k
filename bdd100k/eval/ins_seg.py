"""Evaluation code for BDD100K instance segmentation.

predictions format: BitMasks
"""
import copy
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import (
    DictStrAny,
    NDArrayF64,
    NDArrayI32,
    NDArrayU8,
)
from scalabel.eval.detect import COCOevalV2, DetResult
from scalabel.label.transforms import get_coco_categories
from scalabel.label.typing import Config
from tqdm import tqdm

from ..common.bitmask import bitmask_intersection_rate, parse_bitmask
from ..common.logger import logger
from ..common.utils import reorder_preds


def parse_res_bitmask(
    ann_score: List[Tuple[int, float]], bitmask: NDArrayU8
) -> List[NDArrayI32]:
    """Parse information from result bitmasks and compress its value range."""
    bitmask = bitmask.astype(np.int32)
    category_map = bitmask[:, :, 0]
    ann_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

    ann_ids = []
    scores = []
    category_ids = []

    masks: NDArrayI32 = np.zeros(bitmask.shape[:2], dtype=np.int32)
    i = 0
    ann_score = sorted(ann_score, key=lambda pair: pair[1], reverse=True)
    for ann_id, score in ann_score:
        mask_inds_i = ann_map == ann_id
        if np.count_nonzero(mask_inds_i) == 0:
            continue

        # 0 is for the background
        i += 1
        masks[mask_inds_i] = i
        ann_ids.append(i)
        scores.append(score)

        category_ids_i: NDArrayI32 = np.unique(category_map[mask_inds_i])
        assert category_ids_i.shape[0] == 1
        category_ids.append(category_ids_i[0])

    return [masks, np.array(ann_ids), np.array(scores), np.array(category_ids)]


def get_mask_areas(masks: NDArrayI32) -> NDArrayF64:
    """Get mask areas from the compressed mask map."""
    # 0 for background
    ann_ids = np.sort(np.unique(masks))[1:]
    areas = np.zeros((len(ann_ids)))
    for i, ann_id in enumerate(ann_ids):
        areas[i] = np.count_nonzero(ann_id == masks)
    return areas


class BDD100KInsSegEval(COCOevalV2):
    """Modify the COCO API to support bitmasks as input."""

    def __init__(
        self,
        gt_paths: List[str],
        dt_paths: List[str],
        dt_json: str,
        cat_names: List[str],
        nproc: int = NPROC,
    ) -> None:
        """Initialize InsSeg eval."""
        super().__init__(cat_names)
        self.gt_paths = {os.path.basename(p): p for p in gt_paths}
        self.dt_paths = {os.path.basename(p): p for p in dt_paths}
        self.dt_json = dt_json
        self.nproc = nproc
        self.img_names: List[str] = []
        self.img2score: Dict[str, List[Tuple[int, float]]] = {}
        self.evalImgs: List[DictStrAny] = []
        self.iou_res: List[DictStrAny] = []

        self._prepare()

    def __len__(self) -> int:
        """Get image number."""
        return len(self.img_names)

    def _prepare(self) -> None:
        """Prepare file list for evaluation."""
        self.img_names = list(self.gt_paths.keys())
        self.params.imgIds = self.img_names
        assert len(self.gt_paths) == len(self.dt_paths)
        for img_name in self.img_names:
            assert img_name in self.gt_paths and img_name in self.dt_paths

        with open(self.dt_json, encoding="utf-8") as fp:
            dt_pred = json.load(fp)
        for image in dt_pred:
            img_name = image["name"].replace(".jpg", ".png")
            self.img2score[img_name] = []
            if "labels" not in image or image["labels"] is None:
                continue
            for label in image["labels"]:
                self.img2score[img_name].append(
                    (label["index"], label["score"])
                )
        self.iou_res = [{} for i in range(len(self))]
        if self.nproc > 1:
            with Pool(self.nproc) as pool:
                to_updates: List[DictStrAny] = pool.map(
                    self.compute_iou, tqdm(range(len(self)))
                )
        else:
            to_updates = list(map(self.compute_iou, tqdm(range(len(self)))))
        for res in to_updates:
            self.iou_res[res["ind"]].update(res)

    def evaluate(self) -> None:
        """Run per image evaluation."""
        p = self.params
        p.maxDets = sorted(p.maxDets)
        self.params = p

        # loop through images, area range, max detection number
        if self.nproc > 1:
            with Pool(self.nproc) as pool:
                to_updates: List[Dict[int, DictStrAny]] = pool.map(
                    self.compute_match, range(len(self))
                )
        else:
            to_updates = list(map(self.compute_match, range(len(self))))

        eval_num = len(p.catIds) * len(p.areaRng) * len(self)
        self.evalImgs = [{} for _ in range(eval_num)]
        for to_update in to_updates:
            for ind, item in to_update.items():
                self.evalImgs[ind].update(item)

        self._paramsEval = copy.deepcopy(self.params)

    def compute_iou(self, img_ind: int) -> DictStrAny:
        """Compute IoU per image."""
        img_name = self.img_names[img_ind]
        ann_score = self.img2score[img_name]

        gt_path = self.gt_paths[img_name]
        gt_bitmask: NDArrayU8 = np.asarray(Image.open(gt_path), dtype=np.uint8)
        gt_masks, _, gt_attrs, gt_cat_ids = parse_bitmask(gt_bitmask)
        gt_areas = get_mask_areas(gt_masks)
        gt_crowds = np.bitwise_and(gt_attrs, 2)
        gt_ignores = np.bitwise_and(gt_attrs, 1)

        dt_path = self.dt_paths[img_name]
        dt_bitmask: NDArrayU8 = np.asarray(Image.open(dt_path), dtype=np.uint8)
        dt_masks, _, dt_scores, dt_cat_ids = parse_res_bitmask(
            ann_score, dt_bitmask
        )
        dt_areas = get_mask_areas(dt_masks)

        ious, _ = bitmask_intersection_rate(dt_masks, gt_masks)
        return dict(
            ind=img_ind,
            ious=ious,
            gt_areas=gt_areas,
            gt_cat_ids=gt_cat_ids,
            gt_crowds=gt_crowds,
            gt_ignores=gt_ignores,
            dt_areas=dt_areas,
            dt_scores=dt_scores,
            dt_cat_ids=dt_cat_ids,
        )

    def compute_match(self, img_ind: int) -> Dict[int, DictStrAny]:
        """Compute matching results for each image."""
        res = self.iou_res[img_ind]

        p = self.params
        area_num = len(p.areaRng)
        thr_num = len(p.iouThrs)
        img_num = len(self)

        to_updates = {}
        for cat_ind, cat_id in enumerate(p.catIds):
            gt_inds_c = res["gt_cat_ids"] == cat_id
            gt_areas_c = res["gt_areas"][gt_inds_c]
            gt_crowds_c = res["gt_crowds"][gt_inds_c]
            gt_ignores_c = res["gt_ignores"][gt_inds_c]

            dt_inds_c = res["dt_cat_ids"] == cat_id
            dt_areas_c = res["dt_areas"][dt_inds_c]
            dt_scores_c = res["dt_scores"][dt_inds_c]

            ious_c = res["ious"][dt_inds_c, :][:, gt_inds_c]
            gt_num_c = np.count_nonzero(gt_inds_c)
            dt_num_c = np.count_nonzero(dt_inds_c)

            for area_ind, area_rng in enumerate(p.areaRng):
                gt_matches_a = np.zeros((thr_num, gt_num_c))
                dt_matches_a = np.zeros((thr_num, dt_num_c))
                dt_ignores_a = np.zeros((thr_num, dt_num_c))

                gt_out_of_range_a = np.logical_or(
                    area_rng[0] > gt_areas_c, gt_areas_c > area_rng[1]
                )
                gt_ignores_a = gt_ignores_c | gt_out_of_range_a

                for t_ind, thr in enumerate(p.iouThrs):
                    if ious_c.shape[1] == 0:
                        break
                    ious_t = ious_c.copy()
                    for d_ind in range(ious_t.shape[0]):
                        max_iou = np.max(ious_t[d_ind])
                        g_ind = np.argmax(ious_t[d_ind])
                        if max_iou < thr:
                            continue
                        gt_matches_a[t_ind, g_ind] = 1
                        dt_matches_a[t_ind, d_ind] = 1
                        dt_ignores_a[t_ind, d_ind] = gt_ignores_a[g_ind]
                        if not gt_crowds_c[g_ind]:
                            ious_t[:, g_ind] = 0.0

                dt_out_of_range_a = (
                    np.logical_or(
                        area_rng[0] > dt_areas_c, dt_areas_c > area_rng[1]
                    )
                    .reshape(1, -1)
                    .repeat(thr_num, axis=0)
                )
                dt_ignores_a = np.logical_or(
                    dt_ignores_a,
                    np.logical_and(
                        np.logical_not(dt_matches_a), dt_out_of_range_a
                    ),
                )

                eval_ind: int = (
                    cat_ind * area_num * img_num + area_ind * img_num + img_ind
                )
                to_updates[eval_ind] = dict(
                    category_id=cat_id,
                    aRng=p.areaRng[area_ind],
                    maxDet=p.maxDets[-1],
                    dtMatches=dt_matches_a,
                    gtMatches=gt_matches_a,
                    dtScores=dt_scores_c,
                    gtIgnore=gt_ignores_a,
                    dtIgnore=dt_ignores_a,
                )

        return to_updates


def evaluate_ins_seg(
    gt_paths: List[str],
    pred_paths: List[str],
    pred_score_file: str,
    config: Config,
    nproc: int = NPROC,
    with_logs: bool = True,
) -> DetResult:
    """Load the ground truth and prediction results.

    Args:
        gt_paths: paths to the ground truth bitmasks.
        pred_paths: paths to the prediciton bitmasks.
        pred_score_file: path tothe prediction scores.
        config: Config instance.
        nproc: number of processes.
        with_logs: whether to print logs

    Returns:
        dict: detection metric scores
    """
    categories = get_coco_categories(config)
    cat_ids = [category["id"] for category in categories]
    cat_names = [category["name"] for category in categories]
    pred_paths = reorder_preds(gt_paths, pred_paths)
    bdd_eval = BDD100KInsSegEval(
        gt_paths, pred_paths, pred_score_file, cat_names, nproc
    )
    bdd_eval.params.catIds = cat_ids
    if with_logs:
        logger.info("evaluating...")
    bdd_eval.evaluate()
    if with_logs:
        logger.info("accumulating...")
    bdd_eval.accumulate()
    result = bdd_eval.summarize()  # pylint: disable=redefined-outer-name
    return result
