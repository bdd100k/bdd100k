"""Convert coco to bdd100k format."""
import argparse
from typing import List

from pycocotools.coco import COCO
from scalabel.label.io import save as save_bdd100k
from scalabel.label.typing import Frame as LabeledFrame
from scalabel.label.typing import Label


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to bdd")
    parser.add_argument(
        "--annFile",
        "-a",
        default="/path/to/coco/label/file",
        help="path to coco label file",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        default="/save/path",
        help="path to save bdd formatted label file",
    )
    return parser.parse_args()


def transform(label_file: str) -> List[LabeledFrame]:
    """Transform to bdd100k format."""
    coco = COCO(label_file)
    img_ids = coco.getImgIds()
    img_ids = sorted(img_ids)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    nms = [cat["name"] for cat in cats]
    cat_map = dict(zip(coco.getCatIds(), nms))
    bdd100k_labels = []
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img["id"])
        anns = coco.loadAnns(ann_ids)
        det_dict = LabeledFrame()
        det_dict.name = img["file_name"]
        det_dict.url = img["coco_url"]
        det_dict.attributes = {
            "weather": "undefined",
            "scene": "undefined",
            "timeofday": "undefined",
        }
        det_dict.labels = []
        for i, ann in enumerate(anns):
            label = Label(
                **{
                    "id": ann["id"],
                    "index": i + 1,
                    "category": cat_map[ann["category_id"]],
                    "manualShape": True,
                    "manualAttributes": True,
                    "box_2d": {
                        "x1": ann["bbox"][0],
                        "y1": ann["bbox"][1],
                        "x2": ann["bbox"][0] + ann["bbox"][2] - 1,
                        "y2": ann["bbox"][1] + ann["bbox"][3] - 1,
                    },
                }
            )
            det_dict.labels.append(label)
        bdd100k_labels.append(det_dict)
    return bdd100k_labels


def main() -> None:
    """Main."""
    args = parse_arguments()
    bdd100k_labels = transform(args.annFile)
    save_bdd100k(args.save_path, bdd100k_labels)


if __name__ == "__main__":
    main()
