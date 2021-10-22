"""Test to_scalabel.py."""
import copy
import os
import unittest

from scalabel.label.io import load

from ..common.utils import load_bdd100k_config
from .to_scalabel import IGNORED, bdd100k_to_scalabel


class TestBDD100KToScalabel(unittest.TestCase):
    """Test cases for bdd100k to scalabel conversion."""

    def test_bdd100k_to_scalabel(self) -> None:
        """Test bdd100k_to_scalabel function."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        dataset = load(f"{cur_dir}/testcases/example_ignore_annotation.json")
        frames = dataset.frames
        bdd100k_config = load_bdd100k_config("box_track")
        new_frames = bdd100k_to_scalabel(copy.deepcopy(frames), bdd100k_config)
        self.assertEqual(len(new_frames), 2)
        labels = new_frames[0].labels
        assert labels is not None
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].category, "pedestrian")
        self.assertEqual(labels[1].category, "pedestrian")
        assert labels[0].attributes is not None
        self.assertTrue(labels[0].attributes[IGNORED])
        self.assertEqual(new_frames[1].labels, None)

        bdd100k_config.remove_ignored = True
        new_frames = bdd100k_to_scalabel(frames, bdd100k_config)
        labels = new_frames[0].labels
        assert labels is not None
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].category, "pedestrian")
