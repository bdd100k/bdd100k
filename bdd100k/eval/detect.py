"""Evaluation code for BDD100K detection.

predictions format: List[PredType]
Each predicted bounding box forms one dictionary in BDD100K foramt as follows.
{
    "name": string
    "category": string
    "score": float
    "bbox": [x1, y1, x2, y2]
}
"""
import datetime
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval  # type: ignore
from scalabel.label.io import load as load_bdd100k
from scalabel.label.typing import Frame
from tabulate import tabulate

from ..common.typing import DictAny
from ..common.utils import NAME_MAPPING, read
from ..label.to_coco import bdd100k2coco_det
from .type import GtType, PredType


class COCOV2(COCO):  # type: ignore
    """Modify the COCO API to support annotations dictionary as input."""

    def __init__(
        self,
        annotation_file: Optional[str] = None,
        annotations: Optional[GtType] = None,
    ) -> None:
        """Init."""
        super().__init__(annotation_file)
        # initialize the annotations in COCO format without saving as json.

        if annotation_file is None:
            print("using the loaded annotations")
            assert isinstance(
                annotations, dict
            ), "annotation file format {} not supported".format(
                type(annotations)
            )
            self.dataset = annotations
            self.createIndex()


def evaluate_det(
    ann_file: str,
    pred_file: str,
    out_dir: str = "none",
    ann_format: str = "coco",
) -> Dict[str, float]:
    """Load the ground truth and prediction results.

    Args:
        ann_file: path to the ground truth annotations. "*.json"
        pred_file: path to the prediciton results in BDD format. "*.json"
        out_dir: output_directory
        ann_format: either in `scalabel` format or in `coco` format.

    Returns:
        dict: detection metric scores
    """
    # GT annotations can either in COCO format or in BDD100K format
    # During evaluation, labels under `ignored` class will be ignored.
    if ann_format == "coco":
        coco_gt = COCOV2(ann_file)
        with open(ann_file) as fp:
            ann_coco = json.load(fp)
    else:
        # Convert the annotation file to COCO format
        labels = read(ann_file)
        ann_coco = bdd100k2coco_det(labels)
        coco_gt = COCOV2(None, ann_coco)

    # Load results and convert the predictions
    pred_res = convert_preds(pred_file, ann_coco)
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    cat_names = [cat["name"] for cat in coco_dt.loadCats(cat_ids)]

    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "bbox"
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = img_ids

    return evaluate_workflow(coco_eval, cat_ids, cat_names, out_dir)


def evaluate_workflow(
    coco_eval: COCOeval, cat_ids: List[int], cat_names: List[str], out_dir: str
) -> Dict[str, float]:
    """Execute evaluation."""
    n_tit = 12  # number of evaluation titles
    n_cls = len(cat_ids)  # 10/8 classes for BDD100K detection/tracking
    n_thr = 10  # [.5:.05:.95] T=10 IoU thresholds for evaluation
    n_rec = 101  # [0:.01:1] R=101 recall thresholds for evaluation
    n_area = 4  # A=4 object area ranges for evaluation
    n_mdet = 3  # [1 10 100] M=3 thresholds on max detections per image

    eval_param = {
        "params": {
            "imgIds": [],
            "catIds": [],
            "iouThrs": np.linspace(
                0.5,
                0.95,
                int(np.round((0.95 - 0.5) / 0.05) + 1),
                endpoint=True,
            ).tolist(),
            "recThrs": np.linspace(
                0.0,
                1.00,
                int(np.round((1.00 - 0.0) / 0.01) + 1),
                endpoint=True,
            ).tolist(),
            "maxDets": [1, 10, 100],
            "areaRng": [
                [0 ** 2, 1e5 ** 2],
                [0 ** 2, 32 ** 2],
                [32 ** 2, 96 ** 2],
                [96 ** 2, 1e5 ** 2],
            ],
            "useSegm": 0,
            "useCats": 1,
        },
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ],
        "counts": [n_thr, n_rec, n_cls, n_area, n_mdet],
        "precision": -np.ones(
            (n_thr, n_rec, n_cls, n_area, n_mdet), order="F"
        ),
        "recall": -np.ones((n_thr, n_cls, n_area, n_mdet), order="F"),
    }
    stats_all = -np.ones((n_cls, n_tit))

    for i, (cat_id, cat_name) in enumerate(zip(cat_ids, cat_names)):
        print("\nEvaluate category: %s" % cat_name)
        coco_eval.params.catIds = [cat_id]
        # coco_eval.params.useSegm = ann_type == "segm"
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_all[i, :] = coco_eval.stats
        eval_param["precision"][:, :, i, :, :] = coco_eval.eval[
            "precision"
        ].reshape((n_thr, n_rec, n_area, n_mdet))
        eval_param["recall"][:, i, :, :] = coco_eval.eval["recall"].reshape(
            (n_thr, n_area, n_mdet)
        )

    # Print evaluation results
    stats = np.zeros((n_tit, 1))
    print("\nOverall performance")
    coco_eval.eval = eval_param
    coco_eval.summarize()

    for i in range(n_tit):
        column = stats_all[:, i]
        if len(column > -1) == 0:
            stats[i] = -1
        else:
            stats[i] = np.mean(column[column > -1], axis=0)

    score_titles = [
        "AP",
        "AP_50",
        "AP_75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_max_1",
        "AR_max_10",
        "AR_max_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    scores: Dict[str, float] = {}

    for title, stat in zip(score_titles, stats):
        scores[title] = stat.item()

    if out_dir != "none":
        write_eval(out_dir, scores, eval_param)
    return scores


def write_eval(
    out_dir: str, scores: Dict[str, float], eval_param: DictAny
) -> None:
    """Write the evaluation results to file, print in tabulate format."""
    output_filename = os.path.join(out_dir, "scores.json")
    with open(output_filename, "w") as fp:
        json.dump(scores, fp)

    # print the overall performance in the tabulate format
    print(create_small_table(scores))

    eval_param["precision"] = eval_param["precision"].flatten().tolist()
    eval_param["recall"] = eval_param["recall"].flatten().tolist()

    with open(os.path.join(out_dir, "eval.json"), "w") as fp:
        json.dump(eval_param, fp)


def convert_preds(
    res_file: str, ann_coco: GtType, max_det: int = 100
) -> List[PredType]:
    """Convert the prediction into the coco eval format."""
    imgs_maps = {
        os.path.split(item["file_name"])[-1]: item["id"]
        for item in ann_coco["images"]
    }
    cls_maps = {item["name"]: item["id"] for item in ann_coco["categories"]}

    images: List[Frame] = load_bdd100k(res_file)
    images = sorted(images, key=lambda image: image.name)

    preds: List[PredType] = []
    for image in images:
        image_name = str(image.name)
        labels = sorted(
            image.labels,
            key=lambda label: label.score if label.score else 0.0,
            reverse=True,
        )[:max_det]

        for label in labels:
            if label.category is None:
                continue
            if label.box_2d is None:
                continue
            if label.score is None:
                continue

            cls_name = str(label.category)
            if cls_name in NAME_MAPPING.keys():
                cls_name = NAME_MAPPING[cls_name]
            box2d = label.box_2d
            x1, y1, x2, y2 = box2d.x1, box2d.y1, box2d.x2, box2d.y2

            pred = PredType(
                category=cls_name,
                score=float(label.score),
                name=image_name,
                image_id=imgs_maps[image_name],
                category_id=cls_maps[cls_name],
                bbox=[x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            )
            preds.append(pred)

    return preds


def create_small_table(small_dict: Dict[str, float]) -> str:
    """Create a small table using the keys of small_dict as headers.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values_t = tuple(zip(*small_dict.items()))
    values = ["{:.1f}".format(val * 100) for val in values_t]
    stride = 3
    items: List[Any] = []  # type: ignore
    for i in range(0, len(keys), stride):
        items.append(keys[i : min(i + stride, len(keys))])
        items.append(values[i : min(i + stride, len(keys))])
    table = tabulate(
        items[1:],
        headers=items[0],
        tablefmt="fancy_grid",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table
