"""Test cases for mot.py."""
import os
import unittest

import numpy as np

from .mots import evaluate_mots, mask_intersection_rate


class TestMaskInteractionRate(unittest.TestCase):
    """Test cases for the mask iou/iof computation."""

    def test_mask_interaction_rate(self) -> None:
        """Check mask interaction rate correctness."""
        a_bitmask = np.ones((10, 10), dtype=np.int32)
        a_bitmask[4:, 4:] = 2
        b_bitmask = np.ones((10, 10), dtype=np.int32) * 2
        b_bitmask[:7, :7] = 1

        ious, ioas = mask_intersection_rate(a_bitmask, b_bitmask)
        gt_ious = np.array([[40 / 73, 24 / 91], [9 / 76, 9 / 20]])
        gt_ioas = np.array([[40 / 49, 24 / 51], [9 / 49, 27 / 51]])
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(ious[i, j], gt_ious[i, j])
                self.assertAlmostEqual(ioas[i, j], gt_ioas[i, j])


class TestEvaluteMOTS(unittest.TestCase):
    """Test Cases for BDD100K MOTS evaluation.."""

    def test_mota_motp_idf1(self) -> None:
        """Check MOTP for the MOTS evaluation."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        a_path = "{}/testcases/a.png".format(cur_dir)
        b_path = "{}/testcases/b.png".format(cur_dir)

        gts = [[a_path]]
        results = [[b_path]]

        res = evaluate_mots(gts, results)
        self.assertAlmostEqual(res["pedestrian"]["MOTA"], 2 / 3)
        self.assertAlmostEqual(res["pedestrian"]["MOTP"], 3 / 4)
        self.assertAlmostEqual(res["pedestrian"]["IDF1"], 4 / 5)


if __name__ == "__main__":
    unittest.main()
