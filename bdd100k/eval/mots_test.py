"""Test cases for mot.py."""
import os
import unittest

import numpy as np
from PIL import Image

from .mot import evaluate_track
from .mots import acc_single_video_mots, mask_intersection_rate, parse_bitmasks


class TestMaskIntersectionRate(unittest.TestCase):
    """Test cases for the mask iou/iof computation."""

    def test_mask_intersection_rate(self) -> None:
        """Check mask intersection rate correctness."""
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
        a_path = "{}/testcases/mots/a.png".format(cur_dir)
        b_path = "{}/testcases/mots/b.png".format(cur_dir)

        gts = [[a_path]]
        results = [[b_path]]

        res = evaluate_track(acc_single_video_mots, gts, results)
        self.assertAlmostEqual(res["pedestrian"]["MOTA"], 2 / 3)
        self.assertAlmostEqual(res["pedestrian"]["MOTP"], 3 / 4)
        self.assertAlmostEqual(res["pedestrian"]["IDF1"], 4 / 5)


class TestParseBitmasks(unittest.TestCase):
    """Test Cases for BDD100K MOTS evaluation input parser."""

    def test_parse_bitmasks(self) -> None:
        """Check input parsing for the MOTS evaluation."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bitmask = np.asarray(
            Image.open("{}/testcases/mots/example_bitmask.png".format(cur_dir))
        )
        cvt_maps = parse_bitmasks(bitmask)
        gt_maps = [
            np.load("{}/testcases/mots/gt_{}.npy".format(cur_dir, name))
            for name in ["masks", "ins_ids", "attrs", "cat_ids"]
        ]

        for cvt_map, gt_map in zip(cvt_maps, gt_maps):
            self.assertTrue(np.isclose(cvt_map, gt_map).all())


if __name__ == "__main__":
    unittest.main()
