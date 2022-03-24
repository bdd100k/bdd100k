"""Test cases for evaluation scripts."""
import json
import os
import unittest

import numpy as np
from PIL import Image
from scalabel.common.typing import NDArrayF64, NDArrayU8

from ..common.utils import list_files, load_bdd100k_config
from .ins_seg import evaluate_ins_seg


class TestBDD100KInsSegEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = f"{cur_dir}/testcases/ins_seg/gt"
    pred_base = f"{cur_dir}/testcases/ins_seg/pred"
    pred_json = f"{cur_dir}/testcases/ins_seg/pred.json"
    bdd100k_config = load_bdd100k_config("ins_seg")
    result = evaluate_ins_seg(
        list_files(gt_base, ".png", with_prefix=True),
        list_files(pred_base, ".png", with_prefix=True),
        pred_json,
        bdd100k_config.scalabel,
        nproc=1,
    )

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [
                100.0,
                90.0,
                15.81683168,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                68.60561056,
            ],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [
                68.60561056,
                89.68646865,
                66.66666667,
                68.60561056,
                -1.0,
                -1.0,
                65.83333333,
                70.83333333,
                70.83333333,
                70.83333333,
                -1.0,
                -1.0,
            ],
            dtype=np.float64,
        )
        print(np.nan_to_num(data_arr[-1], nan=-1.0))
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[-1], nan=-1.0), overall_scores
            ).all()
        )

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "AP/pedestrian": 99.99999999999997,
            "AP/rider": 89.99999999999999,
            "AP/car": 15.816831683168317,
            "AP/truck": -1.0,
            "AP/bus": -1.0,
            "AP/train": -1.0,
            "AP/motorcycle": -1.0,
            "AP/bicycle": -1.0,
            "AP": 68.60561056105611,
            "AP50": 89.68646864686468,
            "AP75": 66.66666666666666,
            "APs": 68.60561056105611,
            "APm": -1.0,
            "APl": -1.0,
            "AR1": 65.83333333333333,
            "AR10": 70.83333333333334,
            "AR100": 70.83333333333334,
            "ARs": 70.83333333333334,
            "ARm": -1.0,
            "ARl": -1.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])


def create_test_file() -> None:
    """Creat mocking files for the InsSeg test case."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_base = f"{cur_dir}/testcases/ins_seg/gt"
    dt_base = f"{cur_dir}/testcases/ins_seg/pred"
    dt_json = f"{cur_dir}/testcases/ins_seg/pred.json"

    if not os.path.isdir(gt_base):
        os.makedirs(gt_base)
        gt_mask: NDArrayU8 = np.zeros((100, 100, 4), dtype=np.uint8)
        gt_mask[:10, :10] = np.array([1, 0, 0, 1], dtype=np.uint8)
        gt_mask[20:40, 10:20] = np.array([2, 0, 0, 2], dtype=np.uint8)
        gt_mask[20:40, 20:30] = np.array([3, 0, 0, 3], dtype=np.uint8)
        gt_mask[40:60, 10:30] = np.array([3, 0, 0, 4], dtype=np.uint8)
        gt_mask[40:60, 30:40] = np.array([3, 0, 0, 5], dtype=np.uint8)
        gt_mask[60:70, 50:60] = np.array([3, 0, 0, 6], dtype=np.uint8)
        Image.fromarray(gt_mask).save(os.path.join(gt_base, "a.png"))

    if not os.path.isdir(dt_base):
        os.makedirs(dt_base)
        dt_mask: NDArrayU8 = np.zeros((100, 100, 4), dtype=np.uint8)
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
        with open(dt_json, "w", encoding="utf-8") as fp:
            json.dump(dt_pred, fp)


if __name__ == "__main__":
    unittest.main()
