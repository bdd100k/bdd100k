"""Test cases for evaluation scripts."""
import os
import unittest

from .detect import evaluate_det
from .mot import evaluate_mot
from .run import read


class TestBDD100KEval(unittest.TestCase):
    """Test cases for mot & det BDD100K evaluation."""

    def test_mot(self) -> None:
        """Check mot evaluation correctness."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gts = read("{}/testcases/track_sample_anns/".format(cur_dir))
        preds = read("{}/testcases/track_predictions.json".format(cur_dir))
        result = evaluate_mot(gts, preds)
        overall_reference = {
            "IDF1": 0.7089966679007775,
            "MOTA": 0.6400771952396269,
            "MOTP": 0.8682947680631947,
            "FP": 129,
            "FN": 945,
            "IDSw": 45,
            "MT": 62,
            "PT": 47,
            "ML": 33,
            "FM": 68,
            "mIDF1": 0.3223152925410833,
            "mMOTA": 0.242952917616693,
            "mMOTP": 0.12881014519276474,
        }
        for key in result["OVERALL"]:
            self.assertAlmostEqual(
                result["OVERALL"][key], overall_reference[key]
            )

    def test_det(self) -> None:
        """Check detection evaluation correctness."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gts = "{}/testcases/cocoformat_track_sample_annotations.json".format(
            cur_dir
        )
        preds = "{}/testcases/bbox_predictions.json".format(cur_dir)
        result = evaluate_det(gts, preds)
        overall_reference = {
            "AP": 0.3310833329280097,
            "AP_50": 0.5373587734237398,
            "AP_75": 0.34287281244106843,
            "AP_small": 0.20356852321079935,
            "AP_medium": 0.48831230759261923,
            "AP_large": 0.6425314827066648,
            "AR_max_1": 0.23178269105404029,
            "AR_max_10": 0.3713671493592072,
            "AR_max_100": 0.3993805135329416,
            "AR_small": 0.24934537065196868,
            "AR_medium": 0.5545010044684765,
            "AR_large": 0.6604448777029422,
        }
        for key in result:
            self.assertAlmostEqual(result[key], overall_reference[key])
