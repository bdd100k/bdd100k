"""Test cases for lane.py."""
import os
import unittest
from functools import partial

import numpy as np
from PIL import Image

from .lane import (
    eval_lane_marking,
    eval_lane_per_frame,
    eval_lane_per_threshold,
    get_foreground,
    get_lane_class,
    sub_task_funcs,
)


class TestGetLaneClass(unittest.TestCase):
    """Test cases for the lane specific channel extraction."""

    def test_partialled_classes(self) -> None:
        """Check the function that partial get_lane_class."""
        for num in range(255):
            byte = np.array(num, dtype=np.uint8)
            if num & 8:
                self.assertTrue(get_lane_class(byte, 1, 3, 1))
            else:
                self.assertTrue(get_lane_class(byte, 0, 3, 1))
                self.assertTrue(get_foreground(byte))

            if num & (1 << 5):
                self.assertTrue(sub_task_funcs["direction"](byte, 1))
            else:
                self.assertTrue(sub_task_funcs["direction"](byte, 0))

            if num & (1 << 4):
                self.assertTrue(sub_task_funcs["style"](byte, 1))
            else:
                self.assertTrue(sub_task_funcs["style"](byte, 0))


class TestEvalLanePerThreshold(unittest.TestCase):
    """Test cases for the per image per threshold lane marking evaluation."""

    def test_two_parallel_lines(self) -> None:
        """Check the correctness of the function in general cases."""
        a = np.zeros((10, 10), dtype=np.bool)
        b = np.zeros((10, 10), dtype=np.bool)
        a[3, 3:7] = True
        b[7, 3:7] = True

        for radius in [1, 2, 3]:
            self.assertAlmostEqual(eval_lane_per_threshold(a, b, radius), 0.0)
        for radius in [4, 5, 6]:
            self.assertAlmostEqual(eval_lane_per_threshold(a, b, radius), 1.0)

    def test_two_vertical_lines(self) -> None:
        """Check the correctness of the function in general cases."""
        a = np.zeros((10, 10), dtype=np.bool)
        b = np.zeros((10, 10), dtype=np.bool)
        a[3, 3:6] = True
        b[5:8, 7] = True

        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 2), 0.0)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 3), 1 / 3)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 4), 2 / 3)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 5), 1.0)

    def test_two_vertical_border_lines(self) -> None:
        """Check the correctness of the function in general cases."""
        a = np.zeros((10, 10), dtype=np.bool)
        b = np.zeros((10, 10), dtype=np.bool)
        a[1:6, 1:4] = True
        b[4:7, 3:8] = True

        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 2), 0.0)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 3), 0.4)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 4), 0.70588235)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 5), 1.0)


if __name__ == "__main__":
    unittest.main()
