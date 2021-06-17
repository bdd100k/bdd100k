"""BDD100K lane marking Dataset for pytorch."""

from typing import Tuple

import numpy as np
from PIL import Image
from torch import Tensor

from ..eval.lane import get_foreground, sub_task_cats, sub_task_funcs
from .sem_seg_dataset import BDD100KSemSegDataset


class BDD100KLaneMarkingDataset(BDD100KSemSegDataset):
    """The lane marking dataset for bdd100k.

    During training, the background is treated as 0, the ids for all other
    categories is added by 1.
    """

    _IMAGE_DIR = "images/10k"
    _TARGET_DIR = "labels/lane/masks"
    _TARGET_FILE_EXT = "png"

    def __init__(self, task_name: str, *args, **kwargs) -> None:
        """Init function for the segmentation tracking dataset."""
        super().__init__(*args, **kwargs)
        self.class_func = sub_task_funcs[task_name]
        self.class_num = len(sub_task_cats[task_name])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load image and target given the index."""
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        gt_bytes = np.asarray(target, dtype=np.uint8)
        mask = np.zeros_like(gt_bytes, dtype=np.uint8)
        foreground = get_foreground(gt_bytes)
        for value in range(self.class_num):
            mask_cls = self.class_func(gt_bytes, value) & foreground
            mask = mask * (1 - mask_cls) + (value + 1) * mask_cls
        target = Image.fromarray(mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
