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
            "AP": 0.3129270786276149,
            "AP_50": 0.5208928611619372,
            "AP_75": 0.31352210114424733,
            "AP_small": 0.18365219335000335,
            "AP_medium": 0.4841288233364759,
            "AP_large": 0.6320439295297166,
            "AR_max_1": 0.21599709015331267,
            "AR_max_10": 0.3531322234358392,
            "AR_max_100": 0.3793465642222695,
            "AR_small": 0.22512489216925038,
            "AR_medium": 0.553898980511434,
            "AR_large": 0.6504448777029422,
        }
        for key in result:
            self.assertAlmostEqual(result[key], overall_reference[key])


if __name__ == "__main__":
    unittest.main()
