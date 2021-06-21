"""BDD100K semantic segmentation Dataset for pytorch."""

import os
from typing import Optional, Tuple

from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose


class BDD100KSemSegDataset(VisionDataset):  # type: ignore
    """The semantic segmentation dataset for bdd100k."""

    _IMAGE_DIR = "images/10k"
    _TARGET_DIR = "labels/sem_seg/masks"
    _TARGET_FILE_EXT = "png"

    def __init__(
        self,
        root: str,
        image_set: str,
        list_path: str,
        transform: Optional[Compose] = None,
        target_transform: Optional[Compose] = None,
        transforms: Optional[Compose] = None,
    ) -> None:
        """Init function for the segmentation tracking dataset."""
        super().__init__(root, transforms, transform, target_transform)

        with open(os.path.join(list_path), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(root, self._IMAGE_DIR, image_set)
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(root, self._TARGET_DIR, image_set)
        self.targets = [
            os.path.join(target_dir, x + self._TARGET_FILE_EXT)
            for x in file_names
        ]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load image and target given the index."""
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
