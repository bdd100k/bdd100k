"""Test cases for mot.py."""
import os
import unittest

import numpy as np
from PIL import Image
from scalabel.common.typing import NDArrayF64, NDArrayI32, NDArrayU8

from .bitmask import bitmask_intersection_rate, parse_bitmask


class TestMaskIntersectionRate(unittest.TestCase):
    """Test cases for the mask iou/iof computation."""

    def test_mask_intersection_rate(self) -> None:
        """Check mask intersection rate correctness."""
        a_bitmask: NDArrayI32 = np.ones((10, 10), dtype=np.int32)
        a_bitmask[4:, 4:] = 2
        b_bitmask: NDArrayI32 = np.ones((10, 10), dtype=np.int32)
        b_bitmask *= 2
        b_bitmask[:7, :7] = 1

        ious, ioas = bitmask_intersection_rate(a_bitmask, b_bitmask)
        gt_ious: NDArrayF64 = np.array([[40 / 73, 24 / 91], [9 / 76, 9 / 20]])
        gt_ioas: NDArrayF64 = np.array([[40 / 49, 24 / 51], [9 / 49, 27 / 51]])
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(ious[i, j], gt_ious[i, j])
                self.assertAlmostEqual(ioas[i, j], gt_ioas[i, j])


class TestParseBitmask(unittest.TestCase):
    """Test Cases for the function parse_bitmask."""

    def test_stacked(self) -> None:
        """Check the non-stacked case."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bitmask: NDArrayU8 = np.asarray(
            Image.open(f"{cur_dir}/testcases/example_bitmask.png"),
            dtype=np.uint8,
        )
        cvt_maps = parse_bitmask(bitmask)
        gt_maps = [
            np.load(f"{cur_dir}/testcases/gt_{name}.npy")
            for name in ["masks", "ins_ids", "attrs", "cat_ids"]
        ]

        for cvt_map, gt_map in zip(cvt_maps, gt_maps):
            self.assertTrue(np.isclose(cvt_map, gt_map).all())

    def test_not_stacked(self) -> None:
        """Check the stacked case."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bitmask: NDArrayU8 = np.asarray(
            Image.open(f"{cur_dir}/testcases/example_bitmask.png"),
            dtype=np.uint8,
        )
        cvt_maps = parse_bitmask(bitmask)
        gt_maps = [
            np.load(f"{cur_dir}/testcases/gt_{name}.npy")
            for name in ["stacked_masks", "ins_ids", "attrs", "cat_ids"]
        ]

        for cvt_map, gt_map in zip(cvt_maps, gt_maps):
            self.assertTrue(np.isclose(cvt_map, gt_map).all())


if __name__ == "__main__":
    unittest.main()
