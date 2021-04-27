"""Test cases for to_coco.py."""
import os
import unittest

from ..common.utils import list_files
from .to_coco import (
    bitmask2coco_ins_seg,
    bitmask2coco_seg_track,
    bitmasks_loader,
)

SHAPE = (720, 1280)


class TestBitmasks2COCO(unittest.TestCase):
    """Test Cases for the direct bitmask to coco conversion."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    def test_bitmask_loader(self) -> None:
        """Check the correctness of the bitmask loader."""
        mask_path = "testcases/insseg_bitmask.png"
        instances = bitmasks_loader(mask_path)

        self.assertEqual(len(instances), 4)

        self.assertEqual(instances[0]["instance_id"], 1)
        self.assertEqual(instances[0]["category_id"], 3)
        self.assertFalse(instances[0]["occluded"])
        self.assertEqual(instances[0]["mask"].sum().tolist(), 107995)

        self.assertEqual(instances[-1]["instance_id"], 4)
        self.assertEqual(instances[-1]["category_id"], 3)
        self.assertTrue(instances[-1]["occluded"])
        self.assertEqual(instances[-1]["mask"].sum().tolist(), 2442)

    def test_bitmask2coco_ins_seg(self) -> None:
        """Check the correctness of bitmask2coco_ins_seg."""
        mask_dir = "./testcases"
        coco = bitmask2coco_ins_seg(
            mask_dir,
            SHAPE,
            list_files(mask_dir, suffix="_bitmask.png"),
            [],
        )
        self.assertEqual(len(coco), 4)
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(coco["images"][0]["id"], 1)
        self.assertEqual(coco["images"][0]["file_name"], "insseg_bitmask.png")

    def test_bitmask2coco_seg_track(self) -> None:
        """Check the correctness of bitmask2coco_seg_track."""
        mask_dir = "."
        coco = bitmask2coco_seg_track(
            mask_dir,
            SHAPE,
            list_files(mask_dir, suffix="_bitmask.png"),
            [],
        )
        self.assertEqual(len(coco), 5)
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(coco["images"][0]["id"], 1)
        self.assertEqual(
            coco["images"][0]["file_name"],
            os.path.join("testcases", "insseg_bitmask.png"),
        )

        videos = coco["videos"] if coco["videos"] is not None else []
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["name"], "testcases")


if __name__ == "__main__":
    unittest.main()
