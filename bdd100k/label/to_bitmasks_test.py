"""Test cases for to_bitmasks.py."""
import unittest
import shutil
import os
import json
import numpy as np
from PIL import Image
from .to_bitmasks import segtrack2bitmasks


class TestToBitmasks(unittest.TestCase):
    """Test cases for converting BDD100K labels to bitmasks."""

    def test_conversion(self) -> None:
        """Check conversion to and from bitmask."""
        test_out = './test_bitmasks'

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open(
                "{}/testcases/example_annotation.json".format(cur_dir), "r"
        ) as _file:
            example_labels = [[json.load(_file)]]

        example_bitmasks = [Image.open("{}/testcases/example_bitmask.png"
                                      .format(cur_dir))]

        segtrack2bitmasks(example_labels, test_out, nproc=1)

        # load bitmasks from file
        seq_path = os.path.join(test_out, os.listdir(test_out)[0])
        converted_bitmasks = [np.array(Image.open(os.path.join(seq_path, f)))
                              for f in os.listdir(seq_path)]
        shutil.rmtree(test_out)

        for e, c in zip(example_bitmasks, converted_bitmasks):
            self.assertTrue(np.isclose(e, c).all())


if __name__ == "__main__":
    unittest.main()
