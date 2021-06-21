"""Definition of the BDD100K detection dataset."""

import os
from typing import Dict, List

import numpy as np
from mmdet.datasets import DATASETS, CustomDataset
from scalabel.common.typing import DictStrAny
from scalabel.eval.detect import evaluate_det
from scalabel.label.io import load, save
from scalabel.label.typing import Box2D, Frame, Label
from scalabel.label.utils import get_leaf_categories

from bdd100k.common.utils import (
    check_bdd100k_crowd,
    check_bdd100k_ignored,
    load_bdd100k_config,
)
from bdd100k.label.to_scalabel import bdd100k_to_scalabel

from .typing import AnnInfo, ImgInfo


@DATASETS.register_module()
class BDD100KDetDataset(CustomDataset):  # type: ignore
    """BDD100K Dataset for detecion."""

    def __init__(self, cfg_file: str, nproc: int, *args, **kwargs) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        self.config = load_bdd100k_config(cfg_file)
        self.nproc = nproc
        categories = get_leaf_categories(self.config.scalabel.categories)
        # no `+1` here
        self.cat2label = {
            category.name: i for i, category in enumerate(categories)
        }
        self.CLASSES = [category.name for category in categories]
        self.img_shape = self.config.scalabel.image_size

    def load_annotations(self, ann_file: str) -> List[Frame]:
        """Load annotation form annotation file."""
        frames = load(ann_file, self.nproc).frames
        return bdd100k_to_scalabel(frames, self.config)

    def get_img_info(self, idx: int) -> ImgInfo:
        """Get image information by index."""
        frame: Frame = self.data_infos[idx]
        if self.img_shape is not None:
            height, width = self.img_shape.height, self.img_shape.width
        else:
            assert frame.size is not None
            height, width = frame.size.height, frame.size.width
        return ImgInfo(filename=frame.name, height=height, width=width)

    def get_ann_info(self, idx: int) -> AnnInfo:
        """Get annotation information by index."""
        frame: Frame = self.data_infos[idx]
        assert frame.labels is not None
        gt_bboxes_list = []
        gt_labels_list = []
        gt_bboxes_ignore_list = []
        for label in frame.labels:
            if check_bdd100k_ignored(label):
                continue
            if label.box2d is None:
                continue
            box2d = label.box2d
            if label.category not in self.CLASSES:
                continue
            bbox = [box2d.x1, box2d.y1, box2d.x2, box2d.y2]
            if check_bdd100k_crowd(label):
                gt_bboxes_ignore_list.append(bbox)
            else:
                gt_bboxes_list.append(bbox)
                gt_labels_list.append(self.cat2label[label.category])

        if gt_bboxes_list:
            gt_bboxes = np.array(gt_bboxes_list, dtype=np.float32)
            gt_labels = np.array(gt_labels_list, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore_list:
            gt_bboxes_ignore = np.array(
                gt_bboxes_ignore_list, dtype=np.float32
            )
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = AnnInfo(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore,
        )

        return ann

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get COCO annotation by index."""
        frame: Frame = self.data_infos[idx]
        if frame.labels is not None:
            cat_names = [label.category for label in frame.labels]
        else:
            cat_names = []
        cat_ids = [
            self.cat2label[cat_name]
            for cat_name in cat_names
            if cat_name in self.CLASSES
        ]
        return cat_ids

    def pre_pipeline(self, results: DictStrAny) -> None:
        """Prepare results dict for pipeline."""
        results["img_prefix"] = self.img_prefix
        results["bbox_fields"] = []

    def _filter_imgs(self, min_size: int = 32) -> List[int]:
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, frame in enumerate(self.data_infos):
            if self.filter_empty_gt and (
                frame.labels is None or len(frame.labels) == 0
            ):
                continue
            if frame.size is not None and (
                min(frame.size.height, frame.size.width) >= min_size
            ):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self) -> None:
        """Set flag according to image aspect ratio."""
        self.flag = np.ones(len(self), dtype=np.uint8)

    def prepare_train_img(self, idx: int) -> DictStrAny:
        """Get training data and annotations after pipeline."""
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx: int) -> DictStrAny:
        """Get testing data after pipeline."""
        img_info = self.get_img_info(idx)
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _format_results(  # pylint: disable=arguments-differ
        self, results: List[List[np.ndarray]]
    ) -> List[Frame]:
        """Format the results to the BDD100K prediction format."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results and dataset are not equal: {} != {}".format(
            len(results), len(self)
        )

        frames = []
        ann_id = 0

        for img_idx in range(len(self)):
            img_name = self.data_infos[img_idx]["file_name"]
            frame = Frame(name=img_name, labels=[])
            frames.append(frame)

            result = results[img_idx]
            for cat_idx, bboxes in enumerate(result):
                for bbox in bboxes:
                    box2d = Box2D(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]
                    )
                    ann_id += 1
                    label = Label(
                        id=ann_id,
                        score=bbox[-1],
                        box2d=box2d,
                        category=self.CLASSES[cat_idx],
                    )
                    frame.labels.append(label)  # type: ignore

        return frames

    def format_results(  # pylint: disable=arguments-differ
        self, results: List[List[np.ndarray]], out_dir: str
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        frames = self._format_results(results)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "det.json")
        save(out_path, frames)

    def evaluate(  # pylint: disable=arguments-differ
        self, results: List[List[np.ndarray]]
    ) -> Dict[str, float]:
        """Evaluation in COCO protocol."""
        frames = self._format_results(results)
        scores = evaluate_det(self.data_infos, frames, self.config.scalabel)
        return scores
