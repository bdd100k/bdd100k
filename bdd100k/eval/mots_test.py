"""Test cases for mot.py."""
import os
import unittest

import numpy as np
from scalabel.common.typing import NDArrayF64

from ..common.utils import (
    group_and_sort_files,
    list_files,
    load_bdd100k_config,
)
from .mots import evaluate_seg_track


class TestEvaluteMOTS(unittest.TestCase):
    """Test Cases for BDD100K MOTS evaluation.."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    a_path = f"{cur_dir}/testcases/mots/gt"
    b_path = f"{cur_dir}/testcases/mots/result"

    gts = group_and_sort_files(list_files(a_path, ".png", with_prefix=True))
    results = group_and_sort_files(
        list_files(b_path, ".png", with_prefix=True)
    )
    bdd100k_config = load_bdd100k_config("seg_track")
    result = evaluate_seg_track(
        gts,
        results,
        bdd100k_config.scalabel,
        nproc=1,
    )

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "human",
                "vehicle",
                "bike",
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "AVERAGE",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [
                66.66666667,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                66.66666667,
                -1.0,
                -1.0,
                8.33333333,
                66.66666667,
            ],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [66.66666667, 75.0, 80.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[-1], nan=-1.0), overall_scores
            ).all()
        )

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
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
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in overall_reference.items():
            self.assertAlmostEqual(score, summary[name])
