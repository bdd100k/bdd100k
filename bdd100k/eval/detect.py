"""Evaluation code for BDD100K detection."""
import datetime
import json
import os
from typing import Any, Dict, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bdd100k.label.to_coco import bdd100k2coco_det, bdd100k2coco_track


class COCOV2(COCO):
    """Modify the COCO API to support annotations dictionary as
    input rather than the annotation file."""

    def __init__(
        self,
        annotation_file: Optional[str] = None,
        annotations: Optional[Dict[Any, Any]] = None,
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


def evaluate_det(
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

    # Load results and convert the predictions
    pred_res = convert_preds(pred_file)
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
        print("evaluate category: %s" % (coco_gt.loadCats(cat_id)[0]["name"]))
        coco_eval = COCOeval(coco_gt, coco_dt)
        coco_eval.params.imgIds = img_ids
        coco_eval.params.catIds = coco_dt.getCatIds(catIds=cat_id)
        coco_eval.params.useSegm = ann_type == "segm"
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
    print("overall performance")
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
    scores = []

    for title, stat in zip(score_titles, stats):
        scores.append("{}: {:.3f}".format(title, stat.item()))

    print(scores)
    # output_filename = os.path.join(out_dir, "scores.txt")
    # with open(output_filename, "wb") as fp:
    #     fp.write("\n".join(scores))

    eval_param["precision"] = eval_param["precision"].flatten().tolist()
    eval_param["recall"] = eval_param["recall"].flatten().tolist()

    with open(os.path.join(out_dir, "eval.json"), "w") as fp:
        json.dump(eval_param, fp)


def convert_preds(res_file: str, max_det: int = 100):
    """Convert the prediction into the coco eval format."""
    with open(res_file, "rb") as fp:
        res = json.load(fp)

    # get the list of image_ids in res.
    name = "image_id"
    image_ids = set()
    for item in res:
        if item[name] not in image_ids:
            image_ids.add(item[name])
    image_ids = sorted(list(image_ids))

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
            "Some images have more than {0} detections. Results were cut to {0}"
            " detections per images on {1} images.".format(
                max_det, more_than_max_det
            )
        )

    return res_max_det
