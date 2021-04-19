"""Test cases for to_bitmasks.py."""
import os
import shutil
import unittest

import numpy as np
from PIL import Image
from scalabel.label.io import read

from .to_bitmasks import segtrack_to_bitmasks


class TestToBitmasks(unittest.TestCase):
    """Test cases for converting BDD100K labels to bitmasks."""

    test_out = "./test_bitmasks"

    def test_conversion(self) -> None:
        """Check conversion to and from bitmask."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        labels = read("{}/testcases/example_annotation.json".format(cur_dir))

        example_bitmasks = [
            np.asarray(
                Image.open("{}/testcases/example_bitmask.png".format(cur_dir))
            )
        ]

        segtrack_to_bitmasks(labels, self.test_out, nproc=1)

        # load bitmasks from file
        seq_path = os.path.join(self.test_out, os.listdir(self.test_out)[0])
        converted_bitmasks = [
            np.asarray(Image.open(os.path.join(seq_path, f)))
            for f in os.listdir(seq_path)
        ]

        for e, c in zip(example_bitmasks, converted_bitmasks):
            self.assertTrue((e == c).all())

    @classmethod
    def tearDownClass(cls) -> None:
        """Class teardown for bitmask tests."""
        if os.path.exists(cls.test_out):
            shutil.rmtree(cls.test_out)


if __name__ == "__main__":
    unittest.main()
