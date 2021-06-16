"""BDD100K box tracking Dataset for pytorch."""

import os.path as osp
from random import choice
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from scalabel.label.coco_typing import AnnType
from scalabel.label.io import load, group_and_sort
from scalabel.label.transforms import box2d_to_bbox
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
)
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from ..common.utils import load_bdd100k_config

KEYTYPE = Tuple[int, int, int]


class BDD100KBoxTrackDataset(VisionDataset):  # type: ignore
    """The detection dataset for bdd100k."""

    def __init__(
        self,
        root: str,
        ann_path: str,
        cfg_path: str,
        interval: int = 3,
        nproc: int = 4,
        transform: Optional[Compose] = None,
        target_transform: Optional[Compose] = None,
        transforms: Optional[Compose] = None,
    ) -> None:
        """Init function for the base dataset."""
        super().__init__(root, transforms, transform, target_transform)
        self.interval = interval
        self.config = load_bdd100k_config(cfg_path).scalabel
        self.categories = get_leaf_categories(self.config.categories)
        self.cat2label = {cat.name: i for i, cat in enumerate(self.categories)}
        self.img_shape = self.config.image_size

        frames = load(ann_path, nproc).frames
        self.frames_list = group_and_sort(frames)
        self.index_keys: List[KEYTYPE] = []
        num = 0
        for i, frames in enumerate(self.frames_list):
            for j, _ in enumerate(frames):
                num += 1
                self.index_keys.append((i, j, num))

        global_ins_id = 0
        self.id2ins_id: Dict[str, int] = dict()
        for frame in frames:
            if frame.labels is None:
                continue
            for label in frame.labels:
                id_ = label.id
                if id_ not in self.id2ins_id:
                    global_ins_id += 1
                    self.id2ins_id[id_] = global_ins_id

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.index_keys)

    def _load_image(self, key: KEYTYPE) -> Tensor:
        """Load image given its index."""
        i, j, _ = key
        frame = self.frames_list[i][j]
        path = osp.join(self.root, frame.name)
        img = Image.open(path).convert("RGB")
        return torch.Tensor(img)

    def _load_annos(self, key: KEYTYPE) -> List[AnnType]:
        """Load annotations given its index."""
        annotations: List[AnnType] = []
        i, j, num = key
        frame = self.frames_list[i][j]
        if frame.labels is None:
            return annotations
        for i, label in enumerate(frame.labels):
            if label.box2d is None:
                continue
            if label.category not in self.cat2label:
                continue
            bbox = box2d_to_bbox(label.box2d)  # type: ignore
            annotation = AnnType(
                id=num * 1000 + i + 1,
                image_id=num,
                category_id=self.cat2label[label.category],
                instance_id=self.id2ins_id[label.id],
                iscrowd=int(check_crowd(label) or check_ignored(label)),
                bbox=bbox,
                area=float(bbox[2] * bbox[3]),
            )
            annotations.append(annotation)
        return annotations

    def _load_target(self, key: KEYTYPE) -> Dict[str, Tensor]:
        """Load target given its index."""
        i, j, num = key
        frame = self.frames_list[i][j]
        if self.img_shape is not None:
            img_shape = self.img_shape
        elif frame.size is not None:
            img_shape = frame.size
        else:
            raise ValueError("Image shape not defined!")
        height, width = img_shape.height, img_shape.width

        annos = self._load_annos(key)
        annos = [obj for obj in annos if obj["iscrowd"] == 0]

        classes = [anno["category_id"] for anno in annos]
        iscrowd = [anno["iscrowd"] for anno in annos]

        boxes_ = [obj["bbox"] for obj in annos]
        boxes = torch.Tensor(  # type: ignore
            boxes_, dtype=torch.float  # pylint: disable=no-member
        ).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        area = [obj["area"] for obj in annos]

        target = dict(
            image_id=torch.Tensor([num]),
            labels=torch.LongTensor(classes),  # pylint: disable=no-member
            iscrowd=torch.Tensor(iscrowd),
            boxes=boxes,
            area=torch.Tensor(area),
        )

        return target

    def _sample_reference(self, key: KEYTYPE) -> KEYTYPE:
        """Sample the key of the reference frame."""
        i, j, num = key
        start = max(0, j - self.interval)
        end = min(len(self.frames_list[i]), j + self.interval + 0)
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
