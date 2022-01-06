"""Test cases for lane.py."""
import os
import unittest

import numpy as np
from scalabel.common.typing import NDArrayF64, NDArrayU8

from ..common.utils import list_files
from .lane import (
    eval_lane_per_threshold,
    evaluate_lane_marking,
    get_foreground,
    get_lane_class,
    sub_task_funcs,
)


class TestGetLaneClass(unittest.TestCase):
    """Test cases for the lane specific channel extraction."""

    def test_partialled_classes(self) -> None:
        """Check the function that partial get_lane_class."""
        for num in range(255):
            byte: NDArrayU8 = np.array(num, dtype=np.uint8)
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
        a: NDArrayU8 = np.zeros((10, 10), dtype=bool)
        b: NDArrayU8 = np.zeros((10, 10), dtype=bool)
        a[3, 3:7] = True
        b[7, 3:7] = True

        for radius in [1, 2, 3]:
            self.assertAlmostEqual(eval_lane_per_threshold(a, b, radius), 0.0)
        for radius in [4, 5, 6]:
            self.assertAlmostEqual(eval_lane_per_threshold(a, b, radius), 1.0)

    def test_two_vertical_lines(self) -> None:
        """Check the correctness of the function in general cases."""
        a: NDArrayU8 = np.zeros((10, 10), dtype=bool)
        b: NDArrayU8 = np.zeros((10, 10), dtype=bool)
        a[3, 3:6] = True
        b[5:8, 7] = True

        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 2), 0.0)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 3), 1 / 3)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 4), 2 / 3)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 5), 1.0)


class TestEvaluateLaneMarking(unittest.TestCase):
    """Test cases for the evaluate_lane_marking function."""

    def test_mock_cases(self) -> None:
        """Check the peformance of the mock case."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gt_dir = f"{cur_dir}/testcases/lane/gts"
        res_dir = f"{cur_dir}/testcases/lane/res"
        result = evaluate_lane_marking(
            list_files(gt_dir, ".png", with_prefix=True),
            list_files(res_dir, ".png", with_prefix=True),
            nproc=1,
        )
        data_frame = result.pd_frame()
        data_arr = data_frame.to_numpy()
        gt_data_arr: NDArrayF64 = np.array(
            [
                [70.53328267, 80.9831119, 100.0],
                [100.0, 100.0, 100.0],
                [70.53328267, 80.9831119, 100.0],
                [100.0, 100.0, 100.0],
                [99.82147748, 100.0, 100.0],
                [100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0],
                [75.33066961, 79.34917317, 100.0],
                [71.02916505, 86.25984707, 100.0],
                [100.0, 100.0, 100.0],
                [96.43828133, 100.0, 100.0],
                [94.79621737, 100.0, 100.0],
                [85.26664133, 90.49155595, 100.0],
                [85.26664133, 90.49155595, 100.0],
                [92.17697636, 95.70112753, 100.0],
                [87.57008634, 92.22807981, 100.0],
            ],
            dtype=np.float64,
        )
        data_arr = data_frame.to_numpy()
        self.assertTrue(np.isclose(data_arr, gt_data_arr).all())


if __name__ == "__main__":
    unittest.main()
