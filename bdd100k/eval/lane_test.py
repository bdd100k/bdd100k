"""Test cases for lane.py."""
import os
import unittest
from typing import Dict

import numpy as np

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

    def test_two_vertical_border_lines(self) -> None:
        """Check the correctness of the function in general cases."""
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[1:6, 1:4] = True
        b[4:7, 3:8] = True

        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 2), 0.0)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 3), 0.4)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 4), 0.70588235)
        self.assertAlmostEqual(eval_lane_per_threshold(a, b, 5), 1.0)


class TestEvaluateLaneMarking(unittest.TestCase):
    """Test cases for the evaluate_lane_marking function."""

    def test_mock_cases(self) -> None:
        """Check the peformance of the mock case."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gt_dir = "{}/testcases/lane/gts".format(cur_dir)
        res_dir = "{}/testcases/lane/res".format(cur_dir)
        f_scores = evaluate_lane_marking(gt_dir, res_dir, bound_ths=[1, 2])
        gt_f_scores: Dict[str, float] = {
            "1.0_direction_parallel": 79.46877879291574,
            "2.0_direction_parallel": 87.61816039690531,
            "1.0_direction_vertical": 58.9375575858315,
            "2.0_direction_vertical": 75.23632079381062,
            "1.0_direction_avg": 100.0,
            "2.0_direction_avg": 100.0,
            "1.0_style_solid": 79.46877879291574,
            "2.0_style_solid": 87.61816039690531,
            "1.0_style_dashed": 58.9375575858315,
            "2.0_style_dashed": 75.23632079381062,
            "1.0_style_avg": 100.0,
            "2.0_style_avg": 100.0,
            "1.0_category_crosswalk": 88.24432582570225,
            "2.0_category_crosswalk": 93.82889258902341,
            "1.0_category_double_other": 99.01265721381078,
            "2.0_category_double_other": 100.0,
            "1.0_category_double_white": 100.0,
            "2.0_category_double_white": 100.0,
            "1.0_category_double_yellow": 100.0,
            "2.0_category_double_yellow": 100.0,
            "1.0_category_road_curb": 75.0,
            "2.0_category_road_curb": 75.16008049762166,
            "1.0_category_single_other": 59.173962031069706,
            "2.0_category_single_other": 75.48380881221992,
            "1.0_category_single_white": 100.0,
            "2.0_category_single_white": 100.0,
            "1.0_category_single_yellow": 89.27983318704442,
            "2.0_category_single_yellow": 99.98725140234575,
            "1.0_category_avg": 83.48815417369305,
            "2.0_category_avg": 100.0,
            "1.0_total_avg": 94.496051391231,
            "2.0_total_avg": 100.0,
            "average": 97.2480256956155,
        }
        for key, val in gt_f_scores.items():
            self.assertAlmostEqual(val, f_scores[key])


if __name__ == "__main__":
    unittest.main()
