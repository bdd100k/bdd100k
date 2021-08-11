"""Test cases for lane.py."""
import os
import unittest
from typing import Dict

import numpy as np

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
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[3, 3:7] = True
        b[7, 3:7] = True

        for radius in [1, 2, 3]:
            self.assertAlmostEqual(eval_lane_per_threshold(a, b, radius), 0.0)
        for radius in [4, 5, 6]:
            self.assertAlmostEqual(eval_lane_per_threshold(a, b, radius), 1.0)

    def test_two_vertical_lines(self) -> None:
        """Check the correctness of the function in general cases."""
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
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
        gt_dir = "{}/testcases/lane/gts".format(cur_dir)
        res_dir = "{}/testcases/lane/res".format(cur_dir)
        f_scores = evaluate_lane_marking(
            list_files(gt_dir, ".png", with_prefix=True),
            list_files(res_dir, ".png", with_prefix=True),
            bound_ths=[1, 2],
            nproc=1,
        )
        gt_f_scores: Dict[str, float] = {
            "1.0_direction_parallel": 85.26664133475094,
            "2.0_direction_parallel": 90.4915559511542,
            "1.0_direction_vertical": 70.5332826695019,
            "2.0_direction_vertical": 80.98311190230838,
            "1.0_direction_avg": 100.0,
            "2.0_direction_avg": 100.0,
            "1.0_style_solid": 85.26664133475094,
            "2.0_style_solid": 90.4915559511542,
            "1.0_style_dashed": 70.5332826695019,
            "2.0_style_dashed": 80.98311190230838,
            "1.0_style_avg": 100.0,
            "2.0_style_avg": 100.0,
            "1.0_category_crosswalk": 92.17697635518259,
            "2.0_category_crosswalk": 95.70112753111016,
            "1.0_category_double_other": 99.82147747570991,
            "2.0_category_double_other": 100.0,
            "1.0_category_double_white": 100.0,
            "2.0_category_double_white": 100.0,
            "1.0_category_double_yellow": 100.0,
            "2.0_category_double_yellow": 100.0,
            "1.0_category_road_curb": 75.33066960595205,
            "2.0_category_road_curb": 79.34917317397218,
            "1.0_category_single_other": 71.02916505202546,
            "2.0_category_single_other": 86.25984707490902,
            "1.0_category_single_white": 100.0,
            "2.0_category_single_white": 100.0,
            "1.0_category_single_yellow": 96.43828133485101,
            "2.0_category_single_yellow": 100.0,
            "1.0_category_avg": 94.79621737292219,
            "2.0_category_avg": 100.0,
            "1.0_total_avg": 98.26540579097406,
            "2.0_total_avg": 100.0,
            "average": 99.13270289548703,
        }
        for key, val in gt_f_scores.items():
            self.assertAlmostEqual(val, f_scores[key])


if __name__ == "__main__":
    unittest.main()
