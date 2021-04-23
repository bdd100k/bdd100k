"""Test cases for mot.py."""
import os

import numpy as np

from .seg import evaluate_segmentation, fast_hist


def test_fast_hist() -> None:
    """Check the result of fast_hist."""
    a_bitmask = np.zeros((10, 10), dtype=np.int32)
    a_bitmask[4:, 4:] = 1
    b_bitmask = np.ones((10, 10), dtype=np.int32)
    b_bitmask[:7, :7] = 0

    hist = fast_hist(a_bitmask, b_bitmask, 2)
    gt_hist = np.array([[40, 24], [9, 27]])
    assert np.allclose(hist, gt_hist)


def test_ious_miou() -> None:
    """Check MOTP for the MOTS evaluation."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    a_dir = "{}/testcases/seg/gt".format(cur_dir)
    b_dir = "{}/testcases/seg/pred".format(cur_dir)

    ious = evaluate_segmentation(a_dir, b_dir)
    gt_ious = dict(
        miou=61.73469388,
        road=94.89795918,
        sidewalk=28.57142857,
        wall=0.0,
    )
    for key, val in gt_ious.items():
        assert np.isclose(ious[key], val)
