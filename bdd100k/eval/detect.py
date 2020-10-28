import os
import sys
import numpy as np
import json
import time
from typing import Optional, Dict, Any
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bdd100k.label.to_coco import bdd100k2coco_det


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


def evaluate(ann_file: str, pred_file: str, ann_format: str = "coco") -> Any:
    """Load the ground truth and prediction results.

    Args:
        ann_file: path to the ground truth annotations. "*.json"
        pred_file: path to the prediciton results in BDD format. "*.json"
        ann_format: either in `scalabel` format or in `coco` format.
    """
    if ann_format == "coco":
        coco_gt = COCOV2(ann_file)
    else:
        # Convert the annotation file to COCO format
        with open(ann_file) as fp:
            ann_bdd100k = json.load(fp)
        ann_coco = bdd100k2coco_det(ann_bdd100k)
        coco_gt = COCOV2(None, ann_coco)

    # Load results
    coco_dt = coco_gt.loadRes(pred_file)
