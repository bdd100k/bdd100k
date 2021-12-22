"""Test cases for run.py."""
import argparse
import os
import unittest
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import scalabel.eval.sem_seg as sc_sem_seg
from scalabel.eval.detect import evaluate_det
from scalabel.eval.ins_seg import evaluate_ins_seg as sc_eval_ins_seg
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.mots import (
    acc_single_video_mots,
    evaluate_seg_track as sc_eval_seg_track,
)
from scalabel.eval.pan_seg import evaluate_pan_seg as sc_eval_pan_seg
from scalabel.eval.pose import evaluate_pose
from scalabel.eval.result import Result
from scalabel.label.typing import Dataset, Frame

import bdd100k.eval.run as eval_run

from .run import parse_args, run


def _mock_load(*_):
    return Dataset(
        frames=[
            Frame(
                name="7d06fefd-f7be05a6.png",
                labels=[],
                videoName="7d06fefd-f7be05a6",
                frameIndex=0,
            )
        ]
    )


class TestEvalRun(unittest.TestCase):
    """Test cases for the run function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def mock_rle(self, task: str) -> None:
        """Mock out RLE evaluation functions for testing."""
        def _mock_parse_args(_) -> argparse.Namespace:
            return parse_args(
                [
                    "-t",
                    task,
                    "--config",
                    f"{self.cur_dir}/../configs/sem_seg.toml",
                    "-g",
                    f"{self.cur_dir}/testcases/rle/seg/gt",
                    "-r",
                    f"{self.cur_dir}/testcases/rle/seg/pred",
                ]
            )

        self.parse_args_patch = patch(
            "bdd100k.eval.run.parse_args", _mock_parse_args
        ).__enter__()
        self.load_patch = patch("bdd100k.eval.run.load", _mock_load).__enter__()
        self.sem_seg_patch = patch(
            "bdd100k.eval.run.sc_eval_sem_seg"
        ).__enter__()
        self.ins_seg_patch = patch(
            "bdd100k.eval.run.sc_eval_ins_seg"
        ).__enter__()
        self.seg_track_patch = patch(
            "bdd100k.eval.run.sc_eval_seg_track"
        ).__enter__()
        self.pan_seg_patch = patch(
            "bdd100k.eval.run.sc_eval_pan_seg"
        ).__enter__()

    def mock_bitmask(self, task: str) -> None:
        """Mock out bitmask evaluation functions for testing."""
        def _mock_parse_args(_) -> argparse.Namespace:
            return parse_args(
                [
                    "-t",
                    task,
                    "--config",
                    f"{self.cur_dir}/../configs/sem_seg.toml",
                    "-g",
                    f"{self.cur_dir}/testcases/seg/gt",
                    "-r",
                    f"{self.cur_dir}/testcases/seg/pred",
                ]
            )

        self.parse_args_patch = patch(
            "bdd100k.eval.run.parse_args", _mock_parse_args
        ).__enter__()
        self.load_patch = patch("bdd100k.eval.run.load", _mock_load).__enter__()
        self.sem_seg_patch = patch(
            "bdd100k.eval.run.evaluate_sem_seg"
        ).__enter__()
        self.ins_seg_patch = patch(
            "bdd100k.eval.run.evaluate_ins_seg"
        ).__enter__()
        self.seg_track_patch = patch(
            "bdd100k.eval.run.evaluate_seg_track"
        ).__enter__()
        self.drivable_patch = patch(
            "bdd100k.eval.run.evaluate_drivable"
        ).__enter__()
        self.pan_seg_patch = patch(
            "bdd100k.eval.run.evaluate_pan_seg"
        ).__enter__()

    def test_rle_sem_seg(self) -> None:
        """Test that run calls scalabel sem_seg evaluation."""
        self.mock_rle("sem_seg")
        run()
        self.sem_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_ins_seg(self) -> None:
        """Test that run calls scalabel ins_seg evaluation."""
        self.mock_rle("ins_seg")
        run()
        self.ins_seg_patch.assert_called_once()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_seg_track(self) -> None:
        """Test that run calls scalabel seg_track evaluation."""
        self.mock_rle("seg_track")
        run()
        self.seg_track_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_drivable(self) -> None:
        """Test that run calls scalabel drivable evaluation."""
        self.mock_rle("drivable")
        run()
        self.sem_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_pan_seg(self) -> None:
        """Test that run calls scalabel pan_seg evaluation."""
        self.mock_rle("pan_seg")
        run()
        self.pan_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()

    def test_bitmask_sem_seg(self) -> None:
        """Test that run calls bdd100k sem_seg evaluation."""
        self.mock_bitmask("sem_seg")
        run()
        self.sem_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()
        self.drivable_patch.assert_not_called()

    def test_bitmask_ins_seg(self) -> None:
        """Test that run calls bdd100k ins_seg evaluation."""
        self.mock_bitmask("ins_seg")
        run()
        self.ins_seg_patch.assert_called_once()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()
        self.drivable_patch.assert_not_called()

    def test_bitmask_seg_track(self) -> None:
        """Test that run calls bdd100k seg_track evaluation."""
        self.mock_bitmask("seg_track")
        run()
        self.seg_track_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()
        self.drivable_patch.assert_not_called()

    def test_bitmask_drivable(self) -> None:
        """Test that run calls bdd100k drivable evaluation."""
        self.mock_bitmask("drivable")
        run()
        self.drivable_patch.assert_called_once()
        self.sem_seg_patch.assert_not_called()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_bitmask_pan_seg(self) -> None:
        """Test that run calls bdd100k pan_seg evaluation."""
        self.mock_bitmask("pan_seg")
        run()
        self.pan_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.drivable_patch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
