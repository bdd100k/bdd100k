"""BDD100K segmentation tracking Dataset for pytorch."""

import os.path as osp
from random import choice
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from scalabel.label.coco_typing import AnnType
from scalabel.label.transforms import mask_to_bbox
from scalabel.label.utils import get_leaf_categories
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from .box_track_dataset import KEYTYPE
from ..common.bitmask import parse_bitmasks
from ..common.utils import (
    list_files,
    group_and_sort_files,
    load_bdd100k_config,
)


class BDD100KSegTrackDataset(VisionDataset):  # type: ignore
    """The segmentation tracking dataset for bdd100k."""

    def __init__(
        self,
        root: str,
        ann_path: str,
        cfg_path: str,
        interval: int = 3,
        transform: Optional[Compose] = None,
        target_transform: Optional[Compose] = None,
        transforms: Optional[Compose] = None,
    ) -> None:
        """Init function for the base dataset."""
        super().__init__(root, transforms, transform, target_transform)
        self.interval = interval
        self.ann_path = ann_path
        self.config = load_bdd100k_config(cfg_path).scalabel
        self.categories = get_leaf_categories(self.config.categories)
        self.cat2label = {cat.name: i for i, cat in enumerate(self.categories)}
        self.img_shape = self.config.image_size

        files = list_files(root, suffix=".jpg")
        self.files_list = group_and_sort_files(files)
        self.index_keys: List[KEYTYPE] = []
        num = 0
        for i, frames in enumerate(self.files_list):
            for j, _ in enumerate(frames):
                num += 1
                self.index_keys.append((i, j, num))

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.index_keys)

    def _load_image(self, key: KEYTYPE) -> Tensor:
        """Load image given its index."""
        i, j, _ = key
        name = self.files_list[i][j]
        path = osp.join(self.root, name)
        img = Image.open(path).convert("RGB")
        return torch.Tensor(img)

    def _load_target(self, key: KEYTYPE) -> Dict[str, Tensor]:
        """Load target given its index."""
        i, j, num = key
        name = self.files_list[i][j].replace(".jpg", ".png")
        path = osp.join(self.root, name)
        bitmask = np.asarray(Image.open(path))

        annos: List[AnnType] = []
        masks, ins_ids, attributes, cat_ids = parse_bitmasks(bitmask)
        for mask, ins_id, attribute, cat_id in zip(
            masks, ins_ids, attributes, cat_ids
        ):
            if cat_id >= len(self.categories):
                continue
            anno = AnnType(
                id=num * 1000 + i + 1,
                image_id=num,
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

        annos = [anno for anno in annos if anno["iscrowd"] == 0]

        classes = [anno["category_id"] for anno in annos]
        iscrowd = [anno["iscrowd"] for anno in annos]

        boxes_ = [anno["bbox"] for anno in annos]
        boxes = torch.Tensor(  # type: ignore
            boxes_, dtype=torch.float  # pylint: disable=no-member
        ).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        masks_ = [anno["segmentation"] for anno in annos]
        masks = [
            torch.Tensor(  # type: ignore
                mask, dtype=torch.uint8  # pylint: disable=no-member
            )
            for mask in masks_
        ]
        masks = torch.stack(masks, dim=0)  # pylint: disable=no-member
        area = [anno["area"] for anno in annos]

        target = dict(
            image_id=torch.Tensor([num]),
            labels=torch.LongTensor(classes),  # pylint: disable=no-member
            iscrowd=torch.Tensor(iscrowd),
            boxes=boxes,
            mask=masks,
            area=torch.Tensor(area),
        )

        return target

    def _sample_reference(self, key: KEYTYPE) -> KEYTYPE:
        """Sample the key of the reference frame."""
        i, j, num = key
        start = max(0, j - self.interval)
        end = min(len(self.files_list[i]), j + self.interval + 0)
        k = choice(list(range(start, end)))
        num += k - j
        return (i, k, num)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Load image and target given the index."""
        key = self.index_keys[index]
        ref = self._sample_reference(key)

        key_image = self._load_image(key)
        key_target = self._load_target(key)
        ref_image = self._load_image(ref)
        ref_target = self._load_target(ref)

        if self.transforms is not None:
            key_image, key_target = self.transforms(key_image, key_target)
            ref_image, ref_target = self.transforms(ref_image, ref_target)

        result = dict(
            key_image=key_image,
            ref_image=ref_image,
        )
        for k, v in key_target.items():
            result["key_{}".format(k)] = v
        for k, v in ref_target.items():
            result["ref_{}".format(k)] = v
        return result
