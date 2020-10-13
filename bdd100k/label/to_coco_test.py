"""Test cases for bdd100k2coco.py."""
import json
import os
import unittest

from .to_coco import bdd100k2coco_det


class TestBDD100K2COCO(unittest.TestCase):
    """Test cases for converting BDD100K labels to COCO format."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        "{}/testcases/unitest_val_bdd.json".format(cur_dir), "r"
    ) as _file:
        val_bdd = json.load(_file)

    val_coco = bdd100k2coco_det(val_bdd)

    def test_type(self) -> None:
        """Check coco format type."""
        self.assertTrue(isinstance(self.val_coco, dict))
        self.assertEqual(len(self.val_coco), 4)

    def test_num_images(self) -> None:
        """Check the number of images is unchanged."""
        self.assertEqual(len(self.val_bdd), len(self.val_coco["images"]))

    def test_num_anns(self) -> None:
        """Check the number of annotations is unchanged."""
        len_bdd = sum([len(item["labels"]) for item in self.val_bdd])
        len_coco = len(self.val_coco["annotations"])
        self.assertEqual(len_coco, len_bdd)


if __name__ == "__main__":
    unittest.main()
