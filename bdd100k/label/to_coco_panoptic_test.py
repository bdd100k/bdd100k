"""Test cases for to_bitmasks.py."""
import os
import shutil
import unittest
from typing import Callable, List

import numpy as np
from PIL import Image
from scalabel.label.io import load
from scalabel.label.typing import Config, Frame, Label

from .to_coco_panoptic import (
    bitmask2pan_json,
    bitmask2pan_mask,
    bitmask2coco_panoptic_seg,
)


class TestUtilFunctions(unittest.TestCase):
    """Test case for util function in to_coco_panoptic.py."""

    def test_bitmask2pan_json(self) -> None:
        """Check bitmask to panoptic json file."""
        pass

    def test_bitmask2pan_mask(self) -> None:
        """Check bitmask to panoptic mask png."""
        pass


class TestWholeConversion(unittest.TestCase):
    """Test cases for BDD100K bitmask to coco panoptic segmentation."""

    def test_bitmask2coco_panoptic_seg(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
