"""Evaluation code for BDD100K instance segmentation.

predictions format: BitMasks
"""
import copy
import json
import os
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools.cocoeval import COCOeval  # type: ignore
from scalabel.label.to_coco import load_coco_config
from tqdm import tqdm

from ..common.typing import DictAny
from ..common.utils import list_files
from .detect import evaluate_workflow
from .mots import mask_intersection_rate, parse_bitmasks


def parse_res_bitmasks(
    ann_score: List[Tuple[int, float]], bitmask: np.ndarray
) -> List[np.ndarray]:
    """Parse information from result bitmasks and compress its value range."""
    bitmask = bitmask.astype(np.int32)
    category_map = bitmask[:, :, 0]
    ann_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

    ann_ids = []
    scores = []
    category_ids = []

    masks = np.zeros(bitmask.shape[:2], dtype=np.int32)
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

        category_ids_i = np.unique(category_map[mask_inds_i])
        assert category_ids_i.shape[0] == 1
        category_ids.append(category_ids_i[0])

    ann_ids = np.array(ann_ids)
    scores = np.array(scores)
    category_ids = np.array(category_ids)

    return [masks, ann_ids, scores, category_ids]


def get_mask_areas(masks: np.ndarray) -> np.ndarray:
    """Get mask areas from the compressed mask map."""
    # 0 for background
    ann_ids = np.sort(np.unique(masks))[1:]
    areas = np.zeros((len(ann_ids)))
    for i, ann_id in enumerate(ann_ids):
        areas[i] = np.count_nonzero(ann_id == masks)
    return areas


class BDDInsSegEval(COCOeval):  # type: ignore
    """Modify the COCO API to support bitmasks as input."""

    def __init__(
        self, gt_base: str, dt_base: str, dt_json: str, nproc: int = 4
    ) -> None:
        """Initialize InsSeg eval."""
        super().__init__(iouType="segm")
        self.gt_base = gt_base
        self.dt_base = dt_base
        self.dt_json = dt_json
        self.nproc = nproc
        self.img_names: List[str] = list()
        self.img2score: Dict[str, List[Tuple[int, float]]] = dict()
        self.evalImgs: List[DictAny] = []
        self.iou_res: List[DictAny] = []

        print("Precompute per image IoUs...")
        self._prepare()

    def __len__(self) -> int:
        """Get image number."""
        return len(self.img_names)

    def _prepare(self) -> None:
        """Prepare file list for evaluation."""
        gt_imgs = list_files(self.gt_base, ".png")
        dt_imgs = list_files(self.dt_base, ".png")
        for gt_img, dt_img in zip(gt_imgs, dt_imgs):
            assert gt_img == dt_img
        self.img_names = gt_imgs
        self.params.imgIds = self.img_names  # type: ignore

        with open(self.dt_json) as fp:
            dt_pred = json.load(fp)
        for image in dt_pred:
            self.img2score[image["name"]] = []
            for label in image["labels"]:
                self.img2score[image["name"]].append(
                    (label["index"], label["score"])
                )
        self.iou_res = [dict() for i in range(len(self))]
        with Pool(self.nproc) as pool:
            to_updates: List[DictAny] = pool.map(
                self.compute_iou, tqdm(range(len(self)))
            )
        for res in to_updates:
            self.iou_res[res["ind"]].update(res)

    def evaluate(self) -> None:
        """Run per image evaluation."""
        tic = time.time()

        print("Running per image evaluation...")
        p = self.params  # type: ignore
        eval_num = len(p.catIds) * len(p.areaRng) * len(self)
        self.evalImgs = [dict() for i in range(eval_num)]

        print("Evaluate annotation type *{}*".format(p.iouType))
        p.maxDets = sorted(p.maxDets)

        self.params = p

        # loop through images, area range, max detection number
        with Pool(self.nproc) as pool:
            to_updates: List[Dict[int, DictAny]] = pool.map(
                self.compute_match, range(len(self))
            )
        for to_update in to_updates:
            for ind, item in to_update.items():
                self.evalImgs[ind].update(item)

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def compute_iou(self, img_ind: int) -> DictAny:
        """Compute IoU per image."""
        img_name = self.img_names[img_ind]
        ann_score = self.img2score[img_name]

        gt_path = os.path.join(self.gt_base, img_name)
        gt_bitmask = np.asarray(Image.open(gt_path))
        gt_masks, _, gt_attrs, gt_cat_ids = parse_bitmasks(gt_bitmask)
        gt_areas = get_mask_areas(gt_masks)
        gt_crowds = np.logical_not((gt_attrs & 2).astype(np.bool8))
        gt_ignores = np.logical_not((gt_attrs & 1).astype(np.bool8))

        dt_path = os.path.join(self.dt_base, img_name)
        dt_bitmask = np.asarray(Image.open(dt_path))
        dt_masks, _, dt_scores, dt_cat_ids = parse_res_bitmasks(
            ann_score, dt_bitmask
        )
        dt_areas = get_mask_areas(dt_masks)

        ious, _ = mask_intersection_rate(dt_masks, gt_masks)
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

    def compute_match(self, img_ind: int) -> Dict[int, DictAny]:
        """Compute matching results for each image."""
        res = self.iou_res[img_ind]

        p = self.params
        area_num = len(p.areaRng)
        thr_num = len(p.iouThrs)
        img_num = len(self)

        to_updates = dict()
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
                gt_ignores_a = gt_ignores_c & gt_out_of_range_a

                for t_ind, thr in enumerate(p.iouThrs):
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
    ann_base: str,
    pred_base: str,
    pred_score_file: str,
    cfg_path: str,
    out_dir: str = "none",
) -> Dict[str, float]:
    """Load the ground truth and prediction results.

    Args:
        ann_base: path to the ground truth bitmasks folder.
        pred_base: path to the prediciton bitmasks folder.
        pred_score_file: path tothe prediction scores.
        cfg_path: path to the config file.
        out_dir: output_directory.

    Returns:
        dict: detection metric scores
    """
    categories, _, _ = load_coco_config("ins_seg", cfg_path)
    bdd_eval = BDDInsSegEval(ann_base, pred_base, pred_score_file)
    cat_ids = [int(category["id"]) for category in categories]
    cat_names = [str(category["name"]) for category in categories]
    return evaluate_workflow(bdd_eval, cat_ids, cat_names, out_dir)
