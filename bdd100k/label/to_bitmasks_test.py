"""Test cases for to_bitmasks.py."""
import os
import shutil
import unittest
from typing import Callable, List

import numpy as np
from PIL import Image
from scalabel.label.io import load
from scalabel.label.typing import Frame, Label

from .to_bitmasks import (
    insseg_to_bitmasks,
    segtrack_to_bitmasks,
    semseg_to_bitmasks,
    set_color,
)


class TestUtilFunctions(unittest.TestCase):
    """Test case for util function in to_bitmasks.py."""

    def test_set_color(self) -> None:
        """Check color setting."""
        label = Label(
            id="tmp",
            attributes=dict(truncated=True, crowd=False),
        )
        color = set_color(label, 15, 300, False)
        gt_color = np.array([15, 8, 1, 44])
        self.assertTrue((color == gt_color).all())


class TestToBitmasks(unittest.TestCase):
    """Test cases for converting BDD100K labels to bitmasks."""

    test_out = "./test_bitmasks"

    def task_specific_test(
        self,
        task_name: str,
        output_name: str,
        convert_func: Callable[[List[Frame], str, bool, bool, int], None],
    ) -> None:
        """General test function for different tasks."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        labels = load("{}/testcases/example_annotation.json".format(cur_dir))
        convert_func(labels, self.test_out, False, False, 1)
        output_path = os.path.join(self.test_out, output_name)
        bitmask = np.asarray(Image.open(output_path))

        gt_bitmask = np.asarray(
            Image.open(
                "{}/testcases/{}_bitmask.png".format(cur_dir, task_name)
            )
        )

        self.assertTrue((bitmask == gt_bitmask).all())

    def test_semseg_to_bitmasks(self) -> None:
        """Test case for semantic segmentation to bitmasks."""
        self.task_specific_test(
            "semseg",
            "b1c81faa-3df17267-0000001.png",
            semseg_to_bitmasks,
        )

    def test_insseg_to_bitmasks(self) -> None:
        """Test case for instance segmentation to bitmasks."""
        self.task_specific_test(
            "insseg", "b1c81faa-3df17267-0000001.png", insseg_to_bitmasks
        )

    def test_segtrack_to_bitmasks(self) -> None:
        """Test case for instance segmentation to bitmasks."""
        self.task_specific_test(
            "segtrack",
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
