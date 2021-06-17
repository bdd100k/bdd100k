"""BDD100K instance segmentation Dataset for pytorch."""

from typing import Dict

import torch
from torch import Tensor

from ..label.to_mask import STUFF_NUM
from .ins_seg_dataset import BDD100KInsSegDataset


class BDD100KPanSegDataset(BDD100KInsSegDataset):
    """The panoptic segmentation dataset for bdd100k."""

    def _load_target(self, index: int) -> Dict[str, Tensor]:
        """Load target given its index."""
        target = super()._load_target(index)
        classes = target["classes"]
        isthing = [cls > STUFF_NUM for cls in classes]
        target.update(
            dict(
                isthing=torch.LongTensor(isthing)  # pylint: disable=no-member
            )
        )
        return target
