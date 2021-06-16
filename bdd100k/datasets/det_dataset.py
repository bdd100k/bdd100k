"""BDD100K detection Dataset for pytorch."""

import os.path as osp
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from scalabel.label.coco_typing import AnnType
from scalabel.label.io import load
from scalabel.label.transforms import box2d_to_bbox
from scalabel.label.typing import Frame
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
)
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from ..common.utils import load_bdd100k_config


class BDD100KDetDataset(VisionDataset):  # type: ignore
    """The detection dataset for bdd100k."""

    def __init__(
        self,
        root: str,
        ann_path: str,
        cfg_path: str,
        nproc: int = 4,
        transform: Optional[Compose] = None,
        target_transform: Optional[Compose] = None,
        transforms: Optional[Compose] = None,
    ) -> None:
        """Init function for the base dataset."""
        super().__init__(root, transforms, transform, target_transform)
        self.config = load_bdd100k_config(cfg_path).scalabel
        self.frames = load(ann_path, nproc).frames
        self.categories = get_leaf_categories(self.config.categories)
        self.cat2label = {cat.name: i for i, cat in enumerate(self.categories)}
        self.img_shape = self.config.image_size

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.frames)

    def _load_image(self, index: int) -> Tensor:
        """Load image given its index."""
        frame = self.frames[index]
        path = osp.join(self.root, frame.name)
        img = Image.open(path).convert("RGB")
        return torch.Tensor(img)

    def _load_annos(self, index: int) -> List[AnnType]:
        """Load annotations given its index."""
        annotations: List[AnnType] = []
        frame: Frame = self.frames[index]
        if frame.labels is None:
            return annotations
        for i, label in enumerate(frame.labels):
            if label.box2d is None:
                continue
            if label.category not in self.cat2label:
                continue
            bbox = box2d_to_bbox(label.box2d)  # type: ignore
            annotation = AnnType(
                id=index * 1000 + i + 1,
                image_id=index,
                category_id=self.cat2label[label.category],
                iscrowd=int(check_crowd(label) or check_ignored(label)),
                bbox=bbox,
                area=float(bbox[2] * bbox[3]),
            )
            annotations.append(annotation)
        return annotations

    def _load_target(self, index: int) -> Dict[str, Tensor]:
        """Load target given its index."""
        frame: Frame = self.frames[index]
        if self.img_shape is not None:
            img_shape = self.img_shape
        elif frame.size is not None:
            img_shape = frame.size
        else:
            raise ValueError("Image shape not defined!")
        height, width = img_shape.height, img_shape.width

        annos = self._load_annos(index)
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
            image_id=torch.Tensor([index]),
            labels=torch.LongTensor(classes),  # pylint: disable=no-member
            iscrowd=torch.Tensor(iscrowd),
            boxes=boxes,
            area=torch.Tensor(area),
        )

        return target

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Load image and target given the index."""
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target  # type: ignore
