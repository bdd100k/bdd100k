"""Test cases for run.py."""
import argparse
import os
import unittest
from unittest.mock import MagicMock, patch

from scalabel.label.typing import Dataset, Frame

from .run import run


def mock_load(*_) -> Dataset:  # type: ignore
    """Mock load to return a dummy dataset with one frame."""
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


def construct_args(
    task: str, config: str, ground_truth: str, result: str
) -> argparse.Namespace:
    """Construct command-line arguments for tests."""
    return argparse.Namespace(
        task=task,
        config=config,
        gt=ground_truth,
        result=result,
        nproc=1,
        iou_thr=0.0,
        ignore_iof_thr=False,
        out_file=None,
        score_file="",
        quiet=True,
    )


class TestEvalRun(unittest.TestCase):
    """Test cases for the run function."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parse_args_patch = MagicMock()
    load_patch = MagicMock()
    sem_seg_patch = MagicMock()
    ins_seg_patch = MagicMock()
    seg_track_patch = MagicMock()
    pan_seg_patch = MagicMock()
    drivable_patch = MagicMock()

    def mock_rle(self, task: str) -> None:
        """Mock out RLE evaluation functions for testing."""

        def _mock_parse_args() -> argparse.Namespace:
            return construct_args(
                task=task,
                config=f"{self.cur_dir}/../configs/sem_seg.toml",
                ground_truth=f"{self.cur_dir}/testcases/rle/seg/gt",
                result=f"{self.cur_dir}/testcases/rle/seg/pred",
            )

        patch("bdd100k.eval.run.parse_args", _mock_parse_args).__enter__()
        patch("bdd100k.eval.run.load", mock_load).__enter__()
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

        def _mock_parse_args() -> argparse.Namespace:
            return construct_args(
                task=task,
                config=f"{self.cur_dir}/../configs/sem_seg.toml",
                ground_truth=f"{self.cur_dir}/testcases/seg/gt",
                result=f"{self.cur_dir}/testcases/seg/pred",
            )

        patch("bdd100k.eval.run.parse_args", _mock_parse_args).__enter__()
        patch("bdd100k.eval.run.load", mock_load).__enter__()
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
        """Test that run calls RLE sem_seg evaluation."""
        self.mock_rle("sem_seg")
        run()
        self.sem_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_ins_seg(self) -> None:
        """Test that run calls RLE ins_seg evaluation."""
        self.mock_rle("ins_seg")
        run()
        self.ins_seg_patch.assert_called_once()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_seg_track(self) -> None:
        """Test that run calls RLE seg_track evaluation."""
        self.mock_rle("seg_track")
        run()
        self.seg_track_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_drivable(self) -> None:
        """Test that run calls RLE drivable evaluation."""
        self.mock_rle("drivable")
        run()
        self.sem_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_rle_pan_seg(self) -> None:
        """Test that run calls RLE pan_seg evaluation."""
        self.mock_rle("pan_seg")
        run()
        self.pan_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()

    def test_bitmask_sem_seg(self) -> None:
        """Test that run calls bitmask sem_seg evaluation."""
        self.mock_bitmask("sem_seg")
        run()
        self.sem_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()
        self.drivable_patch.assert_not_called()

    def test_bitmask_ins_seg(self) -> None:
        """Test that run calls bitmask ins_seg evaluation."""
        self.mock_bitmask("ins_seg")
        run()
        self.ins_seg_patch.assert_called_once()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()
        self.drivable_patch.assert_not_called()

    def test_bitmask_seg_track(self) -> None:
        """Test that run calls bitmask seg_track evaluation."""
        self.mock_bitmask("seg_track")
        run()
        self.seg_track_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()
        self.drivable_patch.assert_not_called()

    def test_bitmask_drivable(self) -> None:
        """Test that run calls bitmask drivable evaluation."""
        self.mock_bitmask("drivable")
        run()
        self.drivable_patch.assert_called_once()
        self.sem_seg_patch.assert_not_called()
        self.ins_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.pan_seg_patch.assert_not_called()

    def test_bitmask_pan_seg(self) -> None:
        """Test that run calls bitmask pan_seg evaluation."""
        self.mock_bitmask("pan_seg")
        run()
        self.pan_seg_patch.assert_called_once()
        self.ins_seg_patch.assert_not_called()
        self.sem_seg_patch.assert_not_called()
        self.seg_track_patch.assert_not_called()
        self.drivable_patch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
