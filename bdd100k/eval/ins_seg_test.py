"""Test cases for evaluation scripts."""
import json
import os
import unittest

import numpy as np
from PIL import Image
from scalabel.eval.result import (
    nested_dict_to_data_frame,
    result_to_flatten_dict,
    result_to_nested_dict,
)

from ..common.utils import load_bdd100k_config
from .ins_seg import evaluate_ins_seg


class TestBDD100KInsSegEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = "{}/testcases/ins_seg/gt".format(cur_dir)
    pred_base = "{}/testcases/ins_seg/pred".format(cur_dir)
    pred_json = "{}/testcases/ins_seg/pred.json".format(cur_dir)
    bdd100k_config = load_bdd100k_config("ins_seg")
    result = evaluate_ins_seg(
        gt_base, pred_base, pred_json, bdd100k_config.scalabel, nproc=1
    )
    res_dict = result_to_flatten_dict(result)
    data_frame = nested_dict_to_data_frame(
        result_to_nested_dict(
            result, result._all_classes  # pylint: disable=protected-access
        )
    )

    def test_ins_seg(self) -> None:
        """Check evaluation scores' correctness."""
        overall_reference = {
            "AP": 68.6056105610561,
            "AP50": 89.68646864686468,
            "AP75": 66.66666666666666,
            "APs": 68.6056105610561,
            "APm": 70.92409240924093,
            "APl": 70.92409240924093,
            "AR1": 65.83333333333333,
            "AR10": 70.83333333333334,
            "AR100": 70.83333333333334,
            "ARs": 70.83333333333334,
            "ARm": 70.83333333333334,
            "ARl": 70.83333333333334,
        }
        for key, val in self.res_dict.items():
            self.assertAlmostEqual(val, overall_reference[key])
        self.assertEqual(len(self.res_dict), len(overall_reference))


def create_test_file() -> None:
    """Creat mocking files for the InsSeg test case."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = "{}/testcases/ins_seg/gt".format(cur_dir)
    dt_base = "{}/testcases/ins_seg/pred".format(cur_dir)
    dt_json = "{}/testcases/ins_seg/pred.json".format(cur_dir)

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

    if not os.path.isfile(dt_json):
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
        dt_pred = [
            {
                "name": "a.png",
                "labels": [
                    {
                        "index": item[0],
                        "score": item[1],
                    }
                    for item in scores
                ],
            }
        ]
        with open(dt_json, "w") as fp:
            json.dump(dt_pred, fp)


if __name__ == "__main__":
    unittest.main()
