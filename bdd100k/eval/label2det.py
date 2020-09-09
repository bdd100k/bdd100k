"""Converting label to the detection evaluation format."""
import argparse
import json
from typing import Any, Dict, List

__author__ = "Fisher Yu"
__copyright__ = "Copyright (c) 2018, Fisher Yu"
__email__ = "i@yf.io"
__license__ = "BSD"

DictAny = Dict[str, Any]  # type: ignore[misc]


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("label_path", help="path to the label dir")
    parser.add_argument("det_path", help="path to output detection file")
    args = parser.parse_args()

    return args


def label2det(frames: List[DictAny]) -> List[DictAny]:
    """Converstion."""
    boxes = list()
    for frame in frames:
        for label in frame["labels"]:
            if "box2d" not in label:
                continue
            xy = label["box2d"]
            if xy["x1"] >= xy["x2"] or xy["y1"] >= xy["y2"]:
                continue
            box = {
                "name": frame["name"],
                "timestamp": frame["timestamp"],
                "category": label["category"],
                "bbox": [xy["x1"], xy["y1"], xy["x2"], xy["y2"]],
                "score": 1,
            }
            boxes.append(box)
    return boxes


def convert_labels(label_path: str, det_path: str) -> None:
    """Converation with io."""
    frames = json.load(open(label_path, "r"))
    det = label2det(frames)
    json.dump(det, open(det_path, "w"), indent=4, separators=(",", ": "))


def main() -> None:
    """Main."""
    args = parse_args()
    convert_labels(args.label_path, args.det_path)


if __name__ == "__main__":
    main()
