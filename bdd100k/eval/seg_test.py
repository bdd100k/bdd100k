"""Test cases for mot.py."""
import os
import unittest

import numpy as np
from scalabel.common.typing import NDArrayI32, NDArrayU8

from ..common.utils import list_files
from .seg import evaluate_segmentation, fast_hist, per_image_hist


class TestFastHist(unittest.TestCase):
    """Test cases for the fast_hist function."""

    def test_a_mock_case(self) -> None:
        """Test the correctness for fast_hist."""
        a_bitmask: NDArrayU8 = np.zeros((10, 10), dtype=np.uint8)
        a_bitmask[4:, 4:] = 1
        b_bitmask: NDArrayU8 = np.ones((10, 10), dtype=np.uint8)
        b_bitmask[:7, :7] = 0

        hist = fast_hist(a_bitmask, b_bitmask, 3)[:-1, :-1]
        gt_hist: NDArrayI32 = np.array([[40, 24], [9, 27]], dtype=np.int32)
        self.assertTrue((hist == gt_hist).all())

    def test_pred_overflow(self) -> None:
        """Test the blank prediction overflows."""
        a_bitmask: NDArrayU8 = np.zeros((10, 10), dtype=np.uint8)
        a_bitmask[4:, 4:] = 1
        b_bitmask: NDArrayU8 = np.ones((10, 10), dtype=np.uint8)
        b_bitmask *= 3
        b_bitmask[:7, :7] = 0

        hist = fast_hist(a_bitmask, b_bitmask, 3)[:-1, :-1]
        gt_hist: NDArrayI32 = np.array([[40, 0], [9, 0]], dtype=np.int32)
        self.assertTrue((hist == gt_hist).all())


class TestPerImageHist(unittest.TestCase):
    """Test cases for the per_image_hist function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_general_case(self) -> None:
        """Test the general case."""
        gt_path = f"{self.cur_dir}/testcases/seg/gt/a.png"
        pred_path = f"{self.cur_dir}/testcases/seg/pred/a.png"

        hist, id_set = per_image_hist(gt_path, pred_path, 3)
        gt_hist: NDArrayI32 = np.array([[93, 1], [4, 2]], dtype=np.int32)
        gt_id_set = set([0, 1])
        self.assertTrue((hist[:-1, :-1] == gt_hist).all())
        self.assertSetEqual(id_set, gt_id_set)

    def test_blank_pred(self) -> None:
        """Test the blank prediction case."""
        gt_path = f"{self.cur_dir}/testcases/seg/gt/a.png"
        pred_path = ""

        hist, id_set = per_image_hist(gt_path, pred_path, 3)
        gt_hist: NDArrayI32 = np.array(
            [[0, 0, 94], [0, 0, 6], [0, 0, 0]], dtype=np.int32
        )
        gt_id_set = set([0, 1])
        self.assertTrue((hist == gt_hist).all())
        self.assertSetEqual(id_set, gt_id_set)


class TestEvaluateSegmentation(unittest.TestCase):
    """Test cases for the evaluate_segmentation function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_ious_miou(self) -> None:
        """Test the general case."""
        a_dir = f"{self.cur_dir}/testcases/seg/gt"
        b_dir = f"{self.cur_dir}/testcases/seg/pred"

        result = evaluate_segmentation(
            list_files(a_dir, ".png", with_prefix=True),
            list_files(b_dir, ".png", with_prefix=True),
            nproc=1,
        )
        summary = result.summary()
        gt_summary = {
            "mAcc": 81.27147766323024,
            "mIoU": 61.73469387755102,
            "fIoU": 90.91836734693878,
            "pAcc": 95.0,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])

    def test_blank_dir(self) -> None:
        """Test the missing prediction scenario."""
        a_dir = f"{self.cur_dir}/testcases/seg/gt+"
        b_dir = f"{self.cur_dir}/testcases/seg/pred"

        result = evaluate_segmentation(
            list_files(a_dir, ".png", with_prefix=True),
            list_files(b_dir, ".png", with_prefix=True),
            nproc=1,
        )
        summary = result.summary()
        gt_summary = {
            "mAcc": 81.27147766323024,
            "mIoU": 31.911057692307693,
            "fIoU": 46.45432692307692,
            "pAcc": 47.5,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])
