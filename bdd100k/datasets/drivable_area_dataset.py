"""BDD100K drivable area Dataset for pytorch."""

from .sem_seg_dataset import BDD100KSemSegDataset


class BDD100KDrivableDataset(BDD100KSemSegDataset):
    """The drivable area dataset for bdd100k."""

    _IMAGE_DIR = "images/100k"
    _TARGET_DIR = "labels/sem_seg/masks"
    _TARGET_FILE_EXT = "png"
