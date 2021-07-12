"""Convert coco to bdd100k format."""
import argparse

from scalabel.common.parallel import NPROC
from scalabel.label.from_coco import run


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to bdd100k")
    parser.add_argument(
        "--input",
        "-i",
        help="path to the input coco label file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="path to save bdd formatted label file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for the conversion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
