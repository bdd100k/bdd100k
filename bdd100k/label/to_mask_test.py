"""Test cases for to_bitmasks.py."""
import os
import shutil
import unittest
from typing import Callable, List

import numpy as np
from PIL import Image
from scalabel.label.io import load
from scalabel.label.typing import Frame, Label

from bdd100k.common.typing import BDDConfig

from ..common.typing import BDDConfig
from .to_mask import (
    insseg_to_bitmasks,
    segtrack_to_bitmasks,
    semseg_to_masks,
    set_instance_color,
)


class TestUtilFunctions(unittest.TestCase):
    """Test case for util function in to_bitmasks.py."""

    def test_set_instance_color(self) -> None:
        """Check color setting."""
        label = Label(
            id="tmp",
            attributes=dict(truncated=True, crowd=False),
        )
        color = set_instance_color(label, 15, 300, False)
        gt_color = np.array([15, 8, 1, 44])
        self.assertTrue((color == gt_color).all())


class TestToMasks(unittest.TestCase):
    """Test cases for converting BDD100K labels to masks/bitmasks."""

    test_out = "./test_bitmasks"

    def task_specific_test(
        self,
        file_name: str,
        output_name: str,
        convert_func: Callable[[List[Frame], str, BDDConfig, int], None],
    ) -> None:
        """General test function for different tasks."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        dataset = load("{}/testcases/example_annotation.json".format(cur_dir))
        frames, config = dataset.frames, dataset.config
        convert_func(frames, self.test_out, config, 1)
        output_path = os.path.join(self.test_out, output_name)
        mask = np.asarray(Image.open(output_path))

        gt_mask = np.asarray(
            Image.open("{}/testcases/{}".format(cur_dir, file_name))
        )

        self.assertTrue((mask == gt_mask).all())

    def test_semseg_to_masks(self) -> None:
        """Test case for semantic segmentation to bitmasks."""
        self.task_specific_test(
            "semseg_mask.png",
            "b1c81faa-3df17267-0000001.png",
            semseg_to_masks,
        )

    def test_insseg_to_bitmasks(self) -> None:
        """Test case for instance segmentation to bitmasks."""
        self.task_specific_test(
            "insseg_bitmask.png",
            "b1c81faa-3df17267-0000001.png",
            insseg_to_bitmasks,
        )

    def test_segtrack_to_bitmasks(self) -> None:
        """Test case for instance segmentation to bitmasks."""
        self.task_specific_test(
            "segtrack_bitmask.png",
            "b1c81faa-3df17267/b1c81faa-3df17267-0000001.png",
            segtrack_to_bitmasks,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Class teardown for bitmask tests."""
        if os.path.exists(cls.test_out):
            shutil.rmtree(cls.test_out)


if __name__ == "__main__":
    unittest.main()
