"""Test cases for mot.py."""
import os
import unittest

import numpy as np

from ..common.utils import list_files
from .seg import evaluate_segmentation, fast_hist, per_image_hist


class TestFastHist(unittest.TestCase):
    """Test cases for the fast_hist function."""

    def test_a_mock_case(self) -> None:
        """Test the correctness for fast_hist."""
        a_bitmask = np.zeros((10, 10), dtype=np.uint8)
        a_bitmask[4:, 4:] = 1
        b_bitmask = np.ones((10, 10), dtype=np.uint8)
        b_bitmask[:7, :7] = 0

        hist = fast_hist(a_bitmask, b_bitmask, 2)
        gt_hist = np.array([[40, 24], [9, 27]])
        self.assertTrue((hist == gt_hist).all())

    def test_pred_overflow(self) -> None:
        """Test the blank prediction overflows."""
        a_bitmask = np.zeros((10, 10), dtype=np.uint8)
        a_bitmask[4:, 4:] = 1
        b_bitmask = np.ones((10, 10), dtype=np.uint8)
        b_bitmask *= 2
        b_bitmask[:7, :7] = 0

        hist = fast_hist(a_bitmask, b_bitmask, 2)
        gt_hist = np.array([[40, 24], [9, 27]])
        self.assertTrue((hist == gt_hist).all())


class TestPerImageHist(unittest.TestCase):
    """Test cases for the per_image_hist function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_general_case(self) -> None:
        """Test the general case."""
        gt_path = "{}/testcases/seg/gt/a.png".format(self.cur_dir)
        pred_path = "{}/testcases/seg/pred/a.png".format(self.cur_dir)

        hist, id_set = per_image_hist(gt_path, pred_path, 2)
        gt_hist = np.array([[93, 1], [4, 2]])
        gt_id_set = set([0, 1])
        self.assertTrue((hist == gt_hist).all())
        self.assertSetEqual(id_set, gt_id_set)

    def test_blank_pred(self) -> None:
        """Test the blank prediction case."""
        gt_path = "{}/testcases/seg/gt/a.png".format(self.cur_dir)
        pred_path = ""

        hist, id_set = per_image_hist(gt_path, pred_path, 2)
        gt_hist = np.array([[0, 94], [0, 6]])
        gt_id_set = set([0, 1])
        self.assertTrue((hist == gt_hist).all())
        self.assertSetEqual(id_set, gt_id_set)


class TestEvaluateSegmentation(unittest.TestCase):
    """Test cases for the evaluate_segmentation function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_ious_miou(self) -> None:
        """Test the general case."""
        a_dir = "{}/testcases/seg/gt".format(self.cur_dir)
        b_dir = "{}/testcases/seg/pred".format(self.cur_dir)

        ious = evaluate_segmentation(
            list_files(a_dir, ".png", with_prefix=True),
            list_files(b_dir, ".png", with_prefix=True),
        )
        gt_ious = dict(
            miou=61.73469388,
            road=94.89795918,
            sidewalk=28.57142857,
            wall=0.0,
        )
        for key, val in gt_ious.items():
            self.assertAlmostEqual(val, ious[key])

    def test_blank_dir(self) -> None:
        """Test the missing prediction scenario."""
        a_dir = "{}/testcases/seg/gt+".format(self.cur_dir)
        b_dir = "{}/testcases/seg/pred".format(self.cur_dir)

        ious = evaluate_segmentation(
            list_files(a_dir, ".png", with_prefix=True),
            list_files(b_dir, ".png", with_prefix=True),
        )
        gt_ious = dict(
            miou=31.9110576923,
            road=48.4375,
            sidewalk=15.3846153846,
            wall=0.0,
        )
        for key, val in gt_ious.items():
            self.assertAlmostEqual(val, ious[key])
