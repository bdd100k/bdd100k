"""Define the loading classes."""
import os.path as osp

import numpy as np
from PIL import Image

from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from scalabel.common.typing import DictStrAny
from scalabel.label.transforms import mask_to_bbox

from bdd100k.common.bitmask import parse_bitmask


@PIPELINES.register_module()
class LoadBitMasks:  #  type: ignore
    """Load annotations from BDD100K bitmasks."""

    def __init__(  #  pylint: disable=dangerous-default-value
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_mask: bool = False,
        file_client_args: DictStrAny = dict(backend="disk"),
    ) -> None:
        """Init function."""
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results: DictStrAny) -> DictStrAny:
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        height = results["img_info"]["height"]
        width = results["img_info"]["width"]

        bitmask_path = osp.join(
            results["bitmask_prefix"],
            results["img_info"]["filename"].replace(".jpg", ".png"),
        )
        bitmask = np.asarray(Image.open(bitmask_path))
        masks, ins_ids, attrs, cat_ids = parse_bitmask(bitmask, stacked=True)

        not_crowds = np.logical_not((attrs & 2).astype(bool))
        not_ignored = np.logical_not((attrs & 1).astype(bool))

        masks = masks[not_ignored]
        ins_ids = ins_ids[not_ignored]
        cat_ids = cat_ids[not_ignored]

        results["gt_bboxes"] = [
            mask_to_bbox(mask) for mask in masks[not_crowds]
        ]
        results["gt_bboxes_ingore"] = [
            mask_to_bbox(mask) for mask in masks[np.logical_not(not_crowds)]
        ]
        results["gt_labels"] = [cat_id - 1 for cat_id in cat_ids[not_crowds]]
        results["gt_masks"] = BitmapMasks(
            masks[not_crowds], height=height, width=width
        )
        results["bbox_fields"].append("gt_bboxes")
        results["mask_fields"].append("gt_masks")

        return results

    def __repr__(self) -> str:
        """Repr function."""
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label}, "
        repr_str += f"with_mask={self.with_mask}, "
        return repr_str
