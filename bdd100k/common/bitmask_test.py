"""Test cases for mot.py."""
import os
import unittest

import numpy as np
from PIL import Image
from scalabel.common.typing import NDArrayI32

from .bitmask import bitmask_intersection_rate, parse_bitmask


class TestMaskIntersectionRate(unittest.TestCase):
    """Test cases for the mask iou/iof computation."""

    def test_mask_intersection_rate(self) -> None:
        """Check mask intersection rate correctness."""
        a_bitmask = np.ones((10, 10), dtype=np.int32)
        a_bitmask[4:, 4:] = 2
        b_bitmask: NDArrayI32 = np.ones((10, 10), dtype=np.int32) * 2
        b_bitmask[:7, :7] = 1

        ious, ioas = bitmask_intersection_rate(a_bitmask, b_bitmask)
        gt_ious = np.array([[40 / 73, 24 / 91], [9 / 76, 9 / 20]])
        gt_ioas = np.array([[40 / 49, 24 / 51], [9 / 49, 27 / 51]])
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(ious[i, j], gt_ious[i, j])
                self.assertAlmostEqual(ioas[i, j], gt_ioas[i, j])


class TestParseBitmask(unittest.TestCase):
    """Test Cases for the function parse_bitmask."""

    def test_stacked(self) -> None:
        """Check the non-stacked case."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bitmask = np.asarray(
            Image.open("{}/testcases/example_bitmask.png".format(cur_dir)),
            dtype=np.uint8,
        )
        cvt_maps = parse_bitmask(bitmask)
        gt_maps = [
            np.load("{}/testcases/gt_{}.npy".format(cur_dir, name))
            for name in ["masks", "ins_ids", "attrs", "cat_ids"]
        ]

        for cvt_map, gt_map in zip(cvt_maps, gt_maps):
            self.assertTrue(np.isclose(cvt_map, gt_map).all())

    def test_not_stacked(self) -> None:
        """Check the stacked case."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bitmask = np.asarray(
            Image.open("{}/testcases/example_bitmask.png".format(cur_dir))
        )
        cvt_maps = parse_bitmask(bitmask)
        gt_maps = [
            np.load("{}/testcases/gt_{}.npy".format(cur_dir, name))
            for name in ["stacked_masks", "ins_ids", "attrs", "cat_ids"]
        ]

        for cvt_map, gt_map in zip(cvt_maps, gt_maps):
            self.assertTrue(np.isclose(cvt_map, gt_map).all())


if __name__ == "__main__":
    unittest.main()
