"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np
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
        self.assertAlmostEqual(result["PQ"], 0.6)
        self.assertAlmostEqual(result["SQ"], 0.9)
        self.assertAlmostEqual(result["RQ"], 2 / 3)


class TestEvalPanopticSeg(unittest.TestCase):
    """Test cases for panoptic segmentation's evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = "{}/testcases/pan_seg/gt".format(cur_dir)
    pred_base = "{}/testcases/pan_seg/pred".format(cur_dir)

    def test_pq_per_image(self) -> None:
        """Test case for the function pq_per_image."""
        gt_path = os.path.join(self.gt_base, "a.png")
        pred_path = os.path.join(self.pred_base, "a.png")
        pq_stat = pq_per_image(gt_path, pred_path)
        gt_res_arr = np.array(
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
            ]
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
        res_arr = np.array(res_list)
        self.assertTrue((gt_res_arr == res_arr).all())

    def test_evaluate_pan_seg(self) -> None:
        """Test case for the function evalute_pan_seg."""
        results = evaluate_pan_seg(
            list_files(self.gt_base, suffix=".png", with_prefix=True),
            list_files(self.pred_base, suffix=".png", with_prefix=True),
            nproc=1,
        )
        gt_results = {
            "PQ": 0.6646278879162744,
            "SQ": 0.6835183709387517,
            "RQ": 0.8529411764705882,
            "N": 17,
            "Stuff_PQ": 0.643860238090212,
            "Stuff_SQ": 0.6652694521823528,
            "Stuff_RQ": 0.8333333333333334,
            "Stuff_N": 15,
            "Thing_PQ": 0.8203852616117429,
            "Thing_SQ": 0.8203852616117429,
            "Thing_RQ": 1.0,
            "Thing_N": 2,
        }
        self.assertDictEqual(results, gt_results)


if __name__ == "__main__":
    unittest.main()
