"""Test cases for evaluation scripts."""
import os
import unittest

from .detect import evaluate_det


class TestBDD100KDetectEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

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


if __name__ == "__main__":
    unittest.main()
