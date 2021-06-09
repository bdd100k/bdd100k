"""Test cases for mot.py."""
import os
import unittest

from scalabel.eval.mot import evaluate_track

from ..common.utils import (
    group_and_sort_files,
    list_files,
    load_bdd100k_config,
)
from .mots import acc_single_video_mots


class TestEvaluteMOTS(unittest.TestCase):
    """Test Cases for BDD100K MOTS evaluation.."""

    def test_mota_motp_idf1(self) -> None:
        """Check MOTP for the MOTS evaluation."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        a_path = "{}/testcases/mots/gt".format(cur_dir)
        b_path = "{}/testcases/mots/result".format(cur_dir)

        gts = group_and_sort_files(
            list_files(a_path, ".png", with_prefix=True)
        )
        results = group_and_sort_files(
            list_files(b_path, ".png", with_prefix=True)
        )
        bdd100k_config = load_bdd100k_config("seg_track")
        res = evaluate_track(
            acc_single_video_mots,
            gts,
            results,
            bdd100k_config.scalabel,
            nproc=1,
        )
        self.assertAlmostEqual(res["pedestrian"]["MOTA"], 2 / 3)
        self.assertAlmostEqual(res["pedestrian"]["MOTP"], 3 / 4)
        self.assertAlmostEqual(res["pedestrian"]["IDF1"], 4 / 5)
