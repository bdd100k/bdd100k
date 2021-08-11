"""Test cases for mot.py."""
import os
import unittest

from scalabel.eval.box_track import evaluate_track
from scalabel.eval.result import (
    nested_dict_to_data_frame,
    result_to_flatten_dict,
    result_to_nested_dict,
)

from ..common.utils import (
    group_and_sort_files,
    list_files,
    load_bdd100k_config,
)
from .mots import acc_single_video_mots


class TestEvaluteMOTS(unittest.TestCase):
    """Test Cases for BDD100K MOTS evaluation.."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    a_path = "{}/testcases/mots/gt".format(cur_dir)
    b_path = "{}/testcases/mots/result".format(cur_dir)

    gts = group_and_sort_files(list_files(a_path, ".png", with_prefix=True))
    results = group_and_sort_files(
        list_files(b_path, ".png", with_prefix=True)
    )
    bdd100k_config = load_bdd100k_config("seg_track")
    result = evaluate_track(
        acc_single_video_mots,
        gts,
        results,
        bdd100k_config.scalabel,
        nproc=1,
    )
    res_dict = result_to_flatten_dict(result)
    data_frame = nested_dict_to_data_frame(
        result_to_nested_dict(
            result, result._all_classes  # pylint: disable=protected-access
        )
    )

    def test_result_value(self) -> None:
        """Check evaluation scores' correctness."""
        overall_reference = {
            "mMOTA": 8.333333333333334,
            "mMOTP": 9.375,
            "mIDF1": 10.0,
            "MOTA": 66.66666666666667,
            "MOTP": 75.0,
            "IDF1": 80.0,
            "FP": 0,
            "FN": 1,
            "IDSw": 0,
            "MT": 2,
            "PT": 0,
            "ML": 1,
            "FM": 0,
        }
        for key, val in overall_reference.items():
            self.assertAlmostEqual(self.res_dict[key], val)
        self.assertEqual(len(self.res_dict), len(overall_reference))
