"""Definition of the BDD100K drivable area dataset."""

from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class BDD100KDataset(CustomDataset):  # type: ignore
    """BDD100K dataset for drivable area."""

    CLASSES = ("direct", "alternative", "background")

    PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
