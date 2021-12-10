"""Test cases for to_rle.py."""
import os
import unittest

from scalabel.label.io import load
from scalabel.label.typing import Frame
from scalabel.label.utils import get_leaf_categories

from ..common.utils import load_bdd100k_config
from .to_rle import insseg_to_rle, semseg_to_rle


class TestToRLE(unittest.TestCase):
    """Test Cases for rle conversion."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    def test_insseg_to_rle(self) -> None:
        mask_dir = "./testcases/to_rle/ins_seg/masks"
        score = "./testcases/to_rle/ins_seg/scores.json"
        categories = get_leaf_categories(
            load_bdd100k_config("ins_seg").scalabel.categories
        )
        frame = load(score).frames[0]

        new_frame = insseg_to_rle(frame, mask_dir, categories)

    def test_semseg_to_rle(self) -> None:
        mask_dir = "./testcases/to_rle/sem_seg/masks"
        mask = "0.jpg"
        categories = get_leaf_categories(
            load_bdd100k_config("sem_seg").scalabel.categories
        )
        frame = Frame(name=mask)

        new_frame = semseg_to_rle(frame, mask_dir, categories)


if __name__ == "__main__":
    unittest.main()
