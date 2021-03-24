"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np
from PIL import Image

from .ins_seg import evaluate_ins_seg


class TestBDD100KInsSegEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    def test_ins_seg(self) -> None:
        """Check detection evaluation correctness."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gt_base = "{}/testcases/ins_seg/gt".format(cur_dir)
        pred_base = "{}/testcases/ins_seg/pred".format(cur_dir)
        pred_score_file = "{}/testcases/ins_seg/pred.txt".format(cur_dir)
        result = evaluate_ins_seg(gt_base, pred_base, pred_score_file)
        overall_reference = {
            "AP": 0.686056105610561,
            "AP_50": 0.8968646864686468,
            "AP_75": 0.6666666666666666,
            "AP_small": 0.686056105610561,
            "AR_max_1": 0.6749999999999999,
            "AR_max_10": 0.7083333333333334,
            "AR_max_100": 0.7083333333333334,
            "AR_small": 0.7083333333333334,
        }
        for key in overall_reference:
            self.assertAlmostEqual(result[key], overall_reference[key])


def create_test_file():
    """Creat mocking files for the InsSeg test case."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = "{}/testcases/ins_seg/gt".format(cur_dir)
    dt_base = "{}/testcases/ins_seg/pred".format(cur_dir)
    dt_score_file = "{}/testcases/ins_seg/pred.txt".format(cur_dir)

    if not os.path.isdir(gt_base):
        os.makedirs(gt_base)
        gt_mask = np.zeros((100, 100, 4), dtype=np.uint8)
        gt_mask[:10, :10] = np.array([1, 0, 0, 1], dtype=np.uint8)
        gt_mask[20:40, 10:20] = np.array([2, 0, 0, 2], dtype=np.uint8)
        gt_mask[20:40, 20:30] = np.array([3, 0, 0, 3], dtype=np.uint8)
        gt_mask[40:60, 10:30] = np.array([3, 0, 0, 4], dtype=np.uint8)
        gt_mask[40:60, 30:40] = np.array([3, 0, 0, 5], dtype=np.uint8)
        gt_mask[60:70, 50:60] = np.array([3, 0, 0, 6], dtype=np.uint8)
        Image.fromarray(gt_mask).save(os.path.join(gt_base, "a.png"))

    if not os.path.isdir(dt_base):
        os.makedirs(dt_base)
        dt_mask = np.zeros((100, 100, 4), dtype=np.uint8)
        dt_mask[:10, :10] = np.array([1, 0, 0, 1], dtype=np.uint8)
        dt_mask[20:40, 10:19] = np.array([2, 0, 0, 2], dtype=np.uint8)
        dt_mask[20:40, 20:27] = np.array([3, 0, 0, 4], dtype=np.uint8)
        dt_mask[40:60, 10:22] = np.array([3, 0, 0, 6], dtype=np.uint8)
        dt_mask[40:60, 30:35] = np.array([3, 0, 0, 7], dtype=np.uint8)
        dt_mask[60:70, 50:54] = np.array([3, 0, 0, 8], dtype=np.uint8)
        Image.fromarray(dt_mask).save(os.path.join(dt_base, "a.png"))

    if not os.path.isfile(dt_score_file):
        scores = [
            [1, 0.4],
            [2, 0.9],
            [3, 0.7],
            [4, 0.8],
            [6, 0.9],
            [7, 0.9],
            [8, 0.9],
            [9, 0.9],
        ]
        lines = []
        for score in scores:
            lines.append(
                " ".join(["a.png"] + [str(num) for num in score]) + "\n"
            )
        with open(dt_score_file, "w") as fp:
            fp.writelines(lines)


if __name__ == "__main__":
    create_test_file()
    unittest.main()
