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
from tabulate import tabulate

from bdd100k.eval.type import GtType, PredType
from bdd100k.label.to_coco import bdd100k2coco_det, bdd100k2coco_track
from ..common.typing import DictAny


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
    out_dir: str = "none" ,
    ann_format: str = "coco",
    mode: str = "det",
) -> DictAny:
    """Load the ground truth and prediction results.

    Args:
        ann_file: path to the ground truth annotations. "*.json"
        pred_file: path to the prediciton results in BDD format. "*.json"
        out_dir: output_directory
        ann_format: either in `scalabel` format or in `coco` format.
        mode: `det` or `track` for label conversion.

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
        with open(ann_file) as fp:
            ann_bdd100k = json.load(fp)
        convert_fn = bdd100k2coco_det if mode == "det" else bdd100k2coco_track
        ann_coco = convert_fn(ann_bdd100k)
        coco_gt = COCOV2(None, ann_coco)

    # Load results and convert the predictions
    pred_res = convert_preds(pred_file, ann_coco)
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    n_tit = 12  # number of evaluation titles
    n_cls = len(
        coco_gt.getCatIds()
    )  # 10 classes for BDD100K detection. 8 for classes tracking
    n_thr = 10  # [.5:.05:.95] T=10 IoU thresholds for evaluation
    n_rec = 101  # [0:.01:1] R=101 recall thresholds for evaluation
    n_area = 4  # A=4 object area ranges for evaluation
    n_mdet = 3  # [1 10 100] M=3 thresholds on max detections per image

    stats_all = -np.ones((n_cls, n_tit))

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
    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "bbox"
    for i, cat_id in enumerate(cat_ids):
        print(
            "\nEvaluate category: %s" % (coco_gt.loadCats(cat_id)[0]["name"])
        )
        coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
        coco_eval.params.imgIds = img_ids
        coco_eval.params.catIds = coco_dt.getCatIds(catIds=cat_id)
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
    scores = {}

    for title, stat in zip(score_titles, stats):
        scores[title] = stat.item()

    if out_dir != 'none':
        write_eval(out_dir, scores, eval_param)
    print(scores)
    return scores


def write_eval(out_dir: str, scores: DictAny, eval_param: DictAny) -> None:
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
    with open(res_file, "rb") as fp:
        res = json.load(fp)

    res = pred_to_coco(res, ann_coco)

    # get the list of image_ids in res.
    name = "image_id"
    image_idss = set()
    for item in res:
        if item[name] not in image_idss:
            image_idss.add(item[name])
    image_ids = sorted(list(image_idss))

    # sort res by 'image_id'.
    res = sorted(res, key=lambda k: k["image_id"])

    # get the start and end index in res for each image.
    image_id = image_ids[0]
    idx = 0
    start_end = {}
    for i, res_i in enumerate(res):
        if i == len(res) - 1:
            start_end[image_id] = (idx, i + 1)
        if res_i[name] != image_id:
            start_end[image_id] = (idx, i)
            idx = i
            image_id = res_i[name]

    # cut number of detections to max_det for each image.
    res_max_det = []
    more_than_max_det = 0
    for image_id in image_ids:
        r_img = res[start_end[image_id][0] : start_end[image_id][1]]
        if len(r_img) > max_det:
            more_than_max_det += 1
            r_img = sorted(r_img, key=lambda k: k["score"], reverse=True)[
                :max_det
            ]
        res_max_det.extend(r_img)

    if more_than_max_det > 0:
        print(
            "Some images have more than {0} detections. Results were "
            "cut to {0} detections per images on {1} images.".format(
                max_det, more_than_max_det
            )
        )

    return res_max_det


def pred_to_coco(pred: List[PredType], ann_coco: GtType) -> List[PredType]:
    """Convert the predictions into a compatabile format with COCOAPIs."""
    # update the prediction results
    imgs_maps = {item["file_name"]: item["id"] for item in ann_coco["images"]}
    cls_maps = {item["name"]: item["id"] for item in ann_coco["categories"]}

    # backward compatible replacement
    naming_replacement_dict = {
        "person": "pedestrian",
        "motor": "motorcycle",
        "bike": "bicycle",
    }
    for p in pred:
        # add image_id and category_id
        cls_name: str = p["category"]
        if cls_name in naming_replacement_dict.keys():
            cls_name = naming_replacement_dict[cls_name]
        p["category_id"] = cls_maps[cls_name]
        p["image_id"] = imgs_maps[p["name"]]
        x1, y1, x2, y2 = p["bbox"]  # x1, y1, x2, y2
        p["bbox"] = [x1, y1, x2 - x1, y2 - y1]

    return pred


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
