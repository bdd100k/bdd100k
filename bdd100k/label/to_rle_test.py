"""Test cases for to_rle.py."""
import os
import unittest

from scalabel.label.io import load
from scalabel.label.typing import Frame
from scalabel.label.utils import get_leaf_categories

from ..common.utils import load_bdd100k_config
from .to_rle import insseg_to_rle, segtrack_to_rle, semseg_to_rle


class TestToRLE(unittest.TestCase):
    """Test Cases for rle conversion."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    def test_insseg_to_rle(self) -> None:
        """Test ins_seg to rle conversion."""
        mask_dir = "./testcases/to_rle/ins_seg/masks"
        score = "./testcases/to_rle/ins_seg/scores.json"
        categories = get_leaf_categories(
            load_bdd100k_config("ins_seg").scalabel.categories
        )
        frame = load(score).frames[0]

        new_frame = insseg_to_rle(frame, mask_dir, categories)

        self.assertEqual(len(new_frame.labels), 22)
        for i, label in enumerate(new_frame.labels):
            self.assertEqual(label.score, frame.labels[i].score)

    def test_semseg_to_rle(self) -> None:
        """Test sem_seg to rle conversion."""
        mask_dir = "./testcases/to_rle/sem_seg/masks"
        mask = "0.jpg"
        categories = get_leaf_categories(
            load_bdd100k_config("sem_seg").scalabel.categories
        )
        frame = Frame(name=mask)

        new_frame = semseg_to_rle(frame, mask_dir, categories)

        self.assertEqual(len(new_frame.labels), 11)

    def test_segtrack_to_rle(self) -> None:
        """Test seg_track to rle conversion."""
        mask_dir = "./testcases/to_rle/seg_track/masks"
        masks = ["0/0-1.jpg", "0/0-2.jpg"]
        categories = get_leaf_categories(
            load_bdd100k_config("seg_track").scalabel.categories
        )
        frames = [Frame(name=mask) for mask in masks]

        new_frames = [segtrack_to_rle(frame, mask_dir, categories) for frame in frames]

        self.assertEqual(len(new_frames[0].labels), 10)
        self.assertEqual(new_frames[0].videoName, "0")
        self.assertEqual(new_frames[0].frameIndex, 1)
        self.assertEqual(len(new_frames[1].labels), 10)
        self.assertEqual(new_frames[1].videoName, "0")
        self.assertEqual(new_frames[1].frameIndex, 2)


if __name__ == "__main__":
    unittest.main()
