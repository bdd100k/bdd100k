"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np
from scalabel.common.typing import NDArrayF64
from scalabel.label.coco_typing import PanopticCatType

from ..common.utils import list_files
from .pan_seg import PQStat, evaluate_pan_seg, pq_per_image


class TestPQStat(unittest.TestCase):
    """Test cases for the class PQStat."""

    pq_a = PQStat()
    pq_a.pq_per_cats[1].iou += 0.9
    pq_a.pq_per_cats[1].tp += 1
    pq_b = PQStat()
    pq_b.pq_per_cats[1].fp += 1
    pq_a += pq_b

    def test_iadd(self) -> None:
        """Check the correctness of __iadd__."""
        self.assertEqual(self.pq_a[1].tp, 1)
        self.assertEqual(self.pq_a[1].fp, 1)
        self.assertEqual(self.pq_a[1].iou, 0.9)

    def test_pq_average_zero_case(self) -> None:
        """Check the correctness of pq_averate when n == 0."""
        result = PQStat().pq_average([])
        for val in result.values():
            self.assertAlmostEqual(val, 0)

    def test_pq_average_common_case(self) -> None:
        """Check the correctness of pq_averate when n == 0."""
        category = PanopticCatType(
            id=1, name="", supercategory="", isthing=True, color=[0, 0, 0]
        )
        result = self.pq_a.pq_average([category])
        self.assertAlmostEqual(result["PQ"], 60.0)
        self.assertAlmostEqual(result["SQ"], 90.0)
        self.assertAlmostEqual(result["RQ"], 66.6666666666)


class TestPQPerImage(unittest.TestCase):
    """Test cases for the pq_per_image function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = f"{cur_dir}/testcases/pan_seg/gt"
    pred_base = f"{cur_dir}/testcases/pan_seg/pred"

    def test_general_case(self) -> None:
        """Test a general case."""
        gt_path = os.path.join(self.gt_base, "a.png")
        pred_path = os.path.join(self.pred_base, "a.png")
        pq_stat = pq_per_image(gt_path, pred_path)
        gt_res_arr: NDArrayF64 = np.array(
            [
                [1, 1, 0, 0, 0.7708830548926014],
                [4, 1, 0, 0, 0.6422764227642277],
                [7, 1, 0, 0, 0.9655321433716164],
                [10, 1, 0, 0, 0.9411398791833123],
                [13, 1, 0, 0, 0.5244956772334294],
                [16, 1, 0, 0, 0.5889328063241107],
                [17, 1, 0, 0, 0.6939252336448598],
                [20, 0, 1, 1, 0.0],
                [22, 0, 1, 1, 0.0],
                [26, 1, 0, 0, 0.5384615384615384],
                [27, 1, 0, 0, 0.5962732919254659],
                [28, 1, 0, 0, 0.9368572838007058],
                [29, 1, 0, 0, 0.9642794268243999],
                [30, 1, 0, 0, 0.9662389555811991],
                [34, 1, 0, 0, 0.8457497612225406],
                [35, 4, 0, 0, 3.18008304800378],
            ],
            dtype=np.float64,
        )
        res_list = []
        for key, pq_stat_cat in pq_stat.pq_per_cats.items():
            res_list.append(
                [
                    key,
                    pq_stat_cat.tp,
                    pq_stat_cat.fp,
                    pq_stat_cat.fn,
                    pq_stat_cat.iou,
                ]
            )
        res_arr: NDArrayF64 = np.array(res_list, dtype=np.float64)
        self.assertTrue((gt_res_arr == res_arr).all())

    def test_blank_prediction(self) -> None:
        """Test pq_per_image with blank prediciton."""
        gt_path = os.path.join(self.gt_base, "a.png")
        pred_path = ""
        pq_stat = pq_per_image(gt_path, pred_path)
        for key, pq_stat_cat in pq_stat.pq_per_cats.items():
            self.assertAlmostEqual(pq_stat_cat.iou, 0.0)
            self.assertEqual(pq_stat_cat.tp, 0)
            self.assertEqual(pq_stat_cat.fp, 0)
            if key != 35:
                self.assertEqual(pq_stat_cat.fn, 1)
            else:
                self.assertEqual(pq_stat_cat.fn, 4)


class TestEvalPanopticSeg(unittest.TestCase):
    """Test cases for the evaluate_pan_seg function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = f"{cur_dir}/testcases/pan_seg/gt"
    pred_base = f"{cur_dir}/testcases/pan_seg/pred"

    def test_general_case(self) -> None:
        """Test a general case."""
        result = evaluate_pan_seg(
            list_files(self.gt_base, suffix=".png", with_prefix=True),
            list_files(self.pred_base, suffix=".png", with_prefix=True),
            nproc=1,
        )
        summary = result.summary()

        gt_summary = {
            "PQ": 66.46278879162743,
            "PQ/STUFF": 64.38602380902118,
            "PQ/THING": 82.03852616117427,
            "SQ": 68.35183709387516,
            "SQ/STUFF": 66.52694521823528,
            "SQ/THING": 82.03852616117427,
            "RQ": 85.29411764705883,
            "RQ/STUFF": 83.33333333333333,
            "RQ/THING": 100.0,
            "N": 17,
            "N/STUFF": 15,
            "N/THING": 2,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])

    def test_evaluate_pan_seg(self) -> None:
        """Test for the case that some predictions are missed."""
        gt_base = f"{self.cur_dir}/testcases/pan_seg/gt+"
        result = evaluate_pan_seg(
            list_files(gt_base, suffix=".png", with_prefix=True),
            list_files(self.pred_base, suffix=".png", with_prefix=True),
            nproc=1,
        )
        summary = result.summary()

        gt_summary = {
            "PQ": 49.385236103572474,
            "PQ/STUFF": 48.677620814166644,
            "PQ/THING": 54.69235077411619,
            "SQ": 68.35183709387516,
            "SQ/STUFF": 66.52694521823528,
            "SQ/THING": 82.03852616117427,
            "RQ": 62.35294117647057,
            "RQ/STUFF": 61.77777777777776,
            "RQ/THING": 66.66666666666666,
            "N": 17,
            "N/STUFF": 15,
            "N/THING": 2,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])
