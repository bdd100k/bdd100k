import os
import sys
import numpy as np
import json
import time
import datetime
from typing import Optional, Dict, Any
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bdd100k.label.to_coco import bdd100k2coco_det, bdd100k2coco_track


class COCOV2(COCO):
    """Modify the COCO API to support annotations dictionary as
    input rather than the annotation file."""

    def __init__(
        self,
        annotation_file: Optional[str] = None,
        annotations: Optional[Dict[Any]] = None,
    ) -> None:
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


def evaluate(
    ann_file: str,
    pred_file: str,
    out_dir: str,
    ann_format: str = "coco",
    mode: str = "det",
) -> Any:
    """Load the ground truth and prediction results.

    Args:
        ann_file: path to the ground truth annotations. "*.json"
        pred_file: path to the prediciton results in BDD format. "*.json"
        out_dir: output_directory
        ann_format: either in `scalabel` format or in `coco` format.
        mode: `det` or `track` for label conversion.
    """
    if ann_format == "coco":
        coco_gt = COCOV2(ann_file)
    else:
        # Convert the annotation file to COCO format
        with open(ann_file) as fp:
            ann_bdd100k = json.load(fp)

        convert_fn = bdd100k2coco_det if mode == "det" else bdd100k2coco_track
        ann_coco = convert_fn(ann_bdd100k)
        coco_gt = COCOV2(None, ann_coco)

    # Load results
    coco_dt = coco_gt.loadRes(pred_file)

    stats_all = -np.ones((80, 12))
    catIds = coco_dt.getCatIds()
    T = 10
    R = 101
    K = 80
    A = 4
    M = 3
    eval_param = {
        "params": {
            "imgIds": [],
            "catIds": [],
            "iouThrs": np.linspace(
                0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
            ).tolist(),
            "recThrs": np.linspace(
                0.0, 1.00, np.round((1.00 - 0.0) / 0.01) + 1, endpoint=True
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
        "counts": [T, R, K, A, M],
        "precision": -np.ones((T, R, K, A, M), order="F"),
        "recall": -np.ones((T, K, A, M), order="F"),
    }
    imgIds = sorted(coco_gt.getImgIds())
    catIds = catIds
    annType = "bbox"
    for i, catId in enumerate(catIds):
        print("evaluate category: %s" % (coco_gt.loadCats(catId)[0]["name"]))
        coco_eval = COCOeval(coco_gt, coco_dt)
        params = coco_eval.params
        coco_eval.params.imgIds = imgIds
        coco_eval.params.catIds = coco_dt.getCatIds(catIds=catId)
        coco_eval.params.useSegm = annType == "segm"
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_all[i, :] = coco_eval.stats
        eval_param["precision"][:, :, i, :, :] = coco_eval.eval[
            "precision"
        ].reshape((T, R, A, M))
        eval_param["recall"][:, i, :, :] = coco_eval.eval["recall"].reshape(
            (T, A, M)
        )

    stats = np.zeros((12, 1))
    print("overall performance")
    coco_eval.eval = eval_param
    coco_eval.summarize()

    for i in range(12):
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
    scores = []

    for title, stat in zip(score_titles, stats):
        scores.append("%s: %0.3f" % (title, stat))

    output_filename = os.path.join(out_dir, "scores.txt")
    with open(output_filename, "wb") as fp:
        fp.write("\n".join(scores))

    eval_param["precision"] = eval_param["precision"].flatten().tolist()
    eval_param["recall"] = eval_param["recall"].flatten().tolist()

    with open(os.path.join(out_dir, "eval.json"), "wb") as fp:
        json.dump(eval_param, fp)


def covert_preds():
    """Convert the prediction into the coco eval format."""
    pass