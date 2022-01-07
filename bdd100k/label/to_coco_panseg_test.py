"""Test cases for to_coco_panseg.py."""
import json
import os
import shutil
import unittest

import numpy as np
from PIL import Image
from scalabel.common.typing import NDArrayU8
from scalabel.label.coco_typing import ImgType

from .to_coco_panseg import (
    bitmask2coco_pan_seg,
    bitmask2pan_json,
    bitmask2pan_mask,
)

MASK_NAME = "panseg_mask.png"


class TestToCocoPanSeg(unittest.TestCase):
    """Test case for functions in to_coco_panseg.py."""

    test_out = "./test_to_coco_panseg"
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_bitmask2pan_mask(self) -> None:
        """Check bitmask to panoptic mask png."""
        mask_name = f"{self.cur_dir}/testcases/panseg_bdd100k/{MASK_NAME}"
        gt_pan_name = f"{self.cur_dir}/testcases/panseg_coco/panseg_mask.png"
        pan_name = f"{self.test_out}/{MASK_NAME}"

        os.makedirs(self.test_out, exist_ok=True)
        bitmask2pan_mask(mask_name, pan_name)

        pan_mask: NDArrayU8 = np.asarray(Image.open(pan_name), dtype=np.uint8)
        gt_pan_mask: NDArrayU8 = np.asarray(
            Image.open(gt_pan_name), dtype=np.uint8
        )

        self.assertTrue((pan_mask == gt_pan_mask).all())

    def test_bitmask2pan_json(self) -> None:
        """Check bitmask to panoptic json file."""
        mask_name = f"{self.cur_dir}/testcases/panseg_bdd100k/{MASK_NAME}"
        gt_json_name = f"{self.cur_dir}/testcases/panseg_coco/panseg_coco.json"

        image = ImgType(id=255, file_name="panseg_mask.jpg")
        pan_ann = bitmask2pan_json(image, mask_name)

        with open(gt_json_name, encoding="utf-8") as fp:
            gt_pan_ann = json.load(fp)

        self.assertDictEqual(dict(pan_ann), gt_pan_ann)

    def test_bitmask2coco_pan_seg(self) -> None:
        """Check bitmask dataset to panoptic mask pngs."""
        mask_base = f"{self.cur_dir}/testcases/panseg_bdd100k"
        pan_mask_base = self.test_out

        os.makedirs(self.test_out, exist_ok=True)
        pan_gt = bitmask2coco_pan_seg(mask_base, pan_mask_base, 1)
        self.assertEqual(len(pan_gt), 3)

        categories = pan_gt["categories"]
        self.assertEqual(len(categories), 41)
        self.assertEqual(len(categories[-1]), 5)
        self.assertEqual(categories[-1]["id"], 40)
        self.assertEqual(categories[-1]["name"], "truck")
        self.assertEqual(categories[-1]["supercategory"], "vehicle")
        self.assertTrue(categories[-1]["isthing"])
        self.assertSequenceEqual(categories[-1]["color"], (0, 0, 70))

        images = pan_gt["images"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["id"], 1)
        self.assertEqual(images[0]["file_name"], "panseg_mask.png")

        annotations = pan_gt["annotations"]
        self.assertEqual(len(annotations), 1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Class teardown for bitmask tests."""
        if os.path.exists(cls.test_out):
            shutil.rmtree(cls.test_out)


if __name__ == "__main__":
    unittest.main()
