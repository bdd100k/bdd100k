"""Test cases for from_coco_panseg.py."""
import json
import os
import shutil
import unittest

import numpy as np
from PIL import Image
from scalabel.label.coco_typing import PanopticAnnType

from .from_coco_panseg import panseg2bitmask


class TestFromCocoPanSeg(unittest.TestCase):
    """Test case for functions in to_coco_panseg.py."""

    test_out = "./test_from_coco_panseg"
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_pansseg2bitmask(self) -> None:
        """Check bitmask to panoptic json file."""
        json_name = "{}/testcases/panseg_coco/panseg_coco.json".format(
            self.cur_dir
        )
        with open(json_name, encoding="utf-8") as fp:
            annotation: PanopticAnnType = json.load(fp)
        pan_mask_base = "{}/testcases/panseg_coco".format(self.cur_dir)
        mask_base = self.test_out

        os.makedirs(self.test_out, exist_ok=True)
        panseg2bitmask(annotation, pan_mask_base, mask_base)

        mask_name = "{}/panseg_mask.png".format(self.test_out)
        gt_mask_name = "{}/testcases/panseg_bdd100k/panseg_mask.png".format(
            self.cur_dir
        )
        bitmask = np.asarray(Image.open(mask_name), dtype=np.uint8)
        gt_bitmask = np.asarray(Image.open(gt_mask_name), dtype=np.uint8)
        self.assertTrue((bitmask == gt_bitmask).all())

    @classmethod
    def tearDownClass(cls) -> None:
        """Class teardown for bitmask tests."""
        if os.path.exists(cls.test_out):
            shutil.rmtree(cls.test_out)


if __name__ == "__main__":
    unittest.main()
