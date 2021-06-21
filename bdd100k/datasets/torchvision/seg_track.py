"""BDD100K segmentation tracking Dataset for pytorch."""

import os.path as osp
from random import choice
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from scalabel.label.coco_typing import AnnType
from scalabel.label.transforms import mask_to_bbox
from torch import Tensor

from ...common.bitmask import parse_bitmask
from ...common.utils import group_and_sort_files, list_files
from .box_track import KEYTYPE, BDD100KBaseTrackDataset
from .ins_seg import annotations_to_tensors_seg


class BDD100KSegTrackDataset(BDD100KBaseTrackDataset):
    """The segmentation tracking dataset for bdd100k."""

    def __init__(self, root: str, ann_path: str, *args, **kwargs) -> None:
        """Init function for the box tracking dataset."""
        super().__init__(root, root, ann_path, *args, **kwargs)

        files = list_files(root, suffix=".jpg")
        self.files_list = group_and_sort_files(files)
        index = 0
        for i, frames in enumerate(self.files_list):
            for j, _ in enumerate(frames):
                index += 1
                self.index_keys.append((i, j, index))

    def _load_image(self, key: KEYTYPE) -> Tensor:
        """Load image given its index."""
        i, j, _ = key
        name = self.files_list[i][j]
        path = osp.join(self.root, name)
        img = Image.open(path).convert("RGB")
        return torch.Tensor(img)

    def _load_target(self, key: KEYTYPE) -> Dict[str, Tensor]:
        """Load target given its index."""
        i, j, index = key
        name = self.files_list[i][j].replace(".jpg", ".png")
        path = osp.join(self.root, name)
        bitmask = np.asarray(Image.open(path))

        annos: List[AnnType] = []
        masks, ins_ids, attributes, cat_ids = parse_bitmask(
            bitmask, stacked=True
        )
        for mask, ins_id, attribute, cat_id in zip(
            masks, ins_ids, attributes, cat_ids
        ):
            if cat_id >= len(self.categories):
                continue
            anno = AnnType(
                id=index * 1000 + i + 1,
                image_id=index,
                category_id=cat_id,
                instance_id=ins_id,
                iscrowd=int((attribute & 3).astype(bool)),
                bbox=mask_to_bbox(mask),
                segmentation=mask,
                area=mask.sum(),
            )
            annos.append(anno)

        if self.img_shape is not None:
            height, width = self.img_shape.height, self.img_shape.width
        else:
            height, width = bitmask.shape[:2]

        instance_ids = [anno["instance_id"] for anno in annos]
        target = annotations_to_tensors_seg(annos, index, height, width)
        target.update(
            ids=torch.LongTensor(instance_ids),  # pylint: disable=no-member
        )

        return target

    def _sample_reference(self, key: KEYTYPE) -> KEYTYPE:
        """Sample the key of the reference frame."""
        i, j, index = key
        start = max(0, j - self.interval)
        end = min(len(self.files_list[i]), j + self.interval + 0)
        k = choice(list(range(start, end)))
        index += k - j
        return (i, k, index)
