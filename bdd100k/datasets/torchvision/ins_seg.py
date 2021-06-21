"""BDD100K instance segmentation Dataset for pytorch."""

import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scalabel.label.coco_typing import AnnType
from scalabel.label.transforms import mask_to_bbox
from scalabel.label.utils import get_leaf_categories
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from ...common.bitmask import parse_bitmask
from ...common.utils import list_files, load_bdd100k_config
from .det import annotations_to_tensors


def annotations_to_tensors_seg(
    annos: List[AnnType], index: int, height: int, width: int
) -> Dict[str, Tensor]:
    """Convert a list of annotations to a dict of torchTensor for seg."""
    segm_list = [anno["segmentation"] for anno in annos]
    mask_list = [
        torch.tensor(  # pylint: disable=no-member
            segm, dtype=torch.uint8  # pylint: disable=no-member
        )
        for segm in segm_list
    ]
    masks = torch.stack(mask_list, dim=0)  # pylint: disable=no-member

    target = annotations_to_tensors(annos, index, height, width)
    target.update(masks=masks)
    return target


class BDD100KInsSegDataset(VisionDataset):  # type: ignore
    """The instance segmentation dataset for bdd100k."""

    def __init__(
        self,
        root: str,
        ann_path: str,
        cfg_path: str,
        transform: Optional[Compose] = None,
        target_transform: Optional[Compose] = None,
        transforms: Optional[Compose] = None,
    ) -> None:
        """Init function for the instance segmentation dataset."""
        super().__init__(root, transforms, transform, target_transform)
        self.ann_path = ann_path
        self.config = load_bdd100k_config(cfg_path).scalabel
        self.files = list_files(root, suffix=".jpg")
        self.categories = get_leaf_categories(self.config.categories)
        self.cat2label = {cat.name: i for i, cat in enumerate(self.categories)}
        self.img_shape = self.config.image_size

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load image given its index."""
        name = self.files[index]
        path = osp.join(self.root, name)
        img = Image.open(path).convert("RGB")
        return torch.Tensor(img)

    def _load_target(self, index: int) -> Dict[str, Tensor]:
        """Load target given its index."""
        name = self.files[index].replace(".jpg", ".png")
        path = osp.join(self.root, name)
        bitmask = np.asarray(Image.open(path))

        annos: List[AnnType] = []
        masks, ins_ids, attributes, cat_ids = parse_bitmask(
            bitmask, stacked=True
        )
        i = 0
        for mask, _, attribute, cat_id in zip(
            masks, ins_ids, attributes, cat_ids
        ):
            if cat_id >= len(self.categories):
                continue
            anno = AnnType(
                id=index * 1000 + i + 1,
                image_id=index,
                category_id=cat_id,
                iscrowd=int((attribute & 3).astype(bool)),
                bbox=mask_to_bbox(mask),
                segmentation=mask,
                area=mask.sum(),
            )
            annos.append(anno)
            i += 1

        if self.img_shape is not None:
            height, width = self.img_shape.height, self.img_shape.width
        else:
            height, width = bitmask.shape[:2]

        return annotations_to_tensors_seg(annos, index, height, width)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Load image and target given the index."""
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target  # type: ignore
