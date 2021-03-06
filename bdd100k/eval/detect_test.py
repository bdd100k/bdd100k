"""Test cases for evaluation scripts."""
import os
import unittest

from ..common.utils import DEFAULT_COCO_CONFIG
from .detect import evaluate_det


class TestBDD100KDetectEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    def test_det(self) -> None:
        """Check detection evaluation correctness."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gts = "{}/testcases/track_sample_anns.json".format(cur_dir)
        preds = "{}/testcases/bbox_predictions.json".format(cur_dir)
        result = evaluate_det(gts, preds, DEFAULT_COCO_CONFIG)
        overall_reference = {
            "AP": 0.3402939266840102,
            "AP_50": 0.553239004090397,
            "AP_75": 0.34938733599978766,
            "AP_small": 0.2091624813537171,
            "AP_medium": 0.48436465249988514,
            "AP_large": 0.6428530466767323,
            "AR_max_1": 0.23877338403079265,
            "AR_max_10": 0.3805003678186741,
            "AR_max_100": 0.40951604502247774,
            "AR_small": 0.2555304932799571,
            "AR_medium": 0.5838594871794872,
            "AR_large": 0.6604261954261954,
        }
        for key in result:
            self.assertAlmostEqual(result[key], overall_reference[key])


if __name__ == "__main__":
    unittest.main()
