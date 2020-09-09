"""Convert BDD100K to COCO format."""

import argparse
import json
import os
from typing import Any, Dict

from tqdm import tqdm

DictObject = Dict[str, Any]  # type: ignore[misc]


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="BDD100K to COCO format")
    parser.add_argument(
        "-l",
        "--label_dir",
        default="/path/to/bdd/label/",
        help="root directory of BDD label Json files",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        default="/save/path",
        help="path to save coco formatted label file",
    )
    return parser.parse_args()


def bdd100k2coco_detection(
    attr_dict: DictObject,
    id_dict: Dict[str, int],
    labeled_images: DictObject,
    filename: str,
) -> None:
    """Convert BDD100K format to COCO."""
    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image["file_name"] = i["name"]
        image["height"] = 720
        image["width"] = 1280

        image["id"] = counter

        empty_image = True

        for label in i["labels"]:
            annotation: DictObject = dict()
            if label["category"] in id_dict.keys():
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image["id"]
                x1 = label["box2d"]["x1"]
                y1 = label["box2d"]["y1"]
                x2 = label["box2d"]["x2"]
                y2 = label["box2d"]["y2"]
                annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                annotation["area"] = float((x2 - x1) * (y2 - y1))
                annotation["category_id"] = id_dict[label["category"]]
                annotation["ignore"] = 0
                annotation["id"] = label["id"]
                annotation["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print("saving...")
    json_string = json.dumps(attr_dict)
    with open(filename, "w") as file:
        file.write(json_string)


def main() -> None:
    """Main."""
    args = parse_arguments()

    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "bus"},
        {"supercategory": "none", "id": 5, "name": "truck"},
        {"supercategory": "none", "id": 6, "name": "bike"},
        {"supercategory": "none", "id": 7, "name": "motor"},
        {"supercategory": "none", "id": 8, "name": "traffic light"},
        {"supercategory": "none", "id": 9, "name": "traffic sign"},
        {"supercategory": "none", "id": 10, "name": "train"},
    ]

    attr_id_dict: Dict[str, int] = {
        i["name"]: i["id"] for i in attr_dict["categories"]  # type: ignore
    }

    # create BDD training set detections in COCO format
    print("Loading training set...")
    with open(
        os.path.join(args.label_dir, "bdd100k_labels_images_train.json")
    ) as f:
        train_labels = json.load(f)
    print("Converting training set...")

    out_fn = os.path.join(
        args.save_path, "bdd100k_labels_images_det_coco_train.json"
    )
    bdd100k2coco_detection(attr_dict, attr_id_dict, train_labels, out_fn)

    print("Loading validation set...")
    # create BDD validation set detections in COCO format
    with open(
        os.path.join(args.label_dir, "bdd100k_labels_images_val.json")
    ) as f:
        val_labels = json.load(f)
    print("Converting validation set...")

    out_fn = os.path.join(
        args.save_path, "bdd100k_labels_images_det_coco_val.json"
    )
    bdd100k2coco_detection(attr_dict, attr_id_dict, val_labels, out_fn)


if __name__ == "__main__":
    main()
