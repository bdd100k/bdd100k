"""Test cases for mot.py."""
import os
import unittest

import numpy as np

from .seg import evaluate_segmentation, fast_hist


class TestIoU(unittest.TestCase):
    """Test cases for the segmentation IoU computation."""

    def test_fast_hist(self) -> None:
        """Check the result of fast_hist."""
        a_bitmask = np.zeros((10, 10), dtype=np.int32)
        a_bitmask[4:, 4:] = 1
        b_bitmask = np.ones((10, 10), dtype=np.int32)
        b_bitmask[:7, :7] = 0

        hist = fast_hist(a_bitmask, b_bitmask, 2)
        gt_hist = np.array([[40, 24], [9, 27]])
        for i in range(2):
            for j in range(2):
                self.assertEqual(hist[i, j], gt_hist[i, j])


class TestEvaluteSegmentation(unittest.TestCase):
    """Test Cases for BDD100K Segmentation evaluation.."""

    def test_ious_miou(self) -> None:
        """Check MOTP for the MOTS evaluation."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        a_dir = "{}/testcases/seg/gt".format(cur_dir)
        b_dir = "{}/testcases/seg/pred".format(cur_dir)

        ious, miou = evaluate_segmentation(a_dir, b_dir, 4, 1)
        gt_ious = [97.47474747, 37.5, 0.0, 50.0]
        for iou, gt_iou in zip(ious, gt_ious):
            self.assertAlmostEqual(iou, gt_iou)
        self.assertAlmostEqual(miou, 46.243686868686865)


if __name__ == "__main__":
    unittest.main()
