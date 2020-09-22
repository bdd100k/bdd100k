"""Convert BDD100K to COCO format."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from tqdm import tqdm

DictObject = Dict[str, Any]  # type: ignore[misc]


def parse_arguments() -> argparse.Namespace:
    """arguments."""
    parser = argparse.ArgumentParser(description="BDD100K to COCO format")
    parser.add_argument(
        "-i",
        "--in-path",
        default="/input/path/",
        help="path to detection JSON file or tracking base folder",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        default="/output/path",
        help="path to save coco formatted label file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["det", "track"],
        help="conversion mode: detection or tracking",
    )
    return parser.parse_args()


def bdd100k2coco_det(labels: List[DictObject]) -> DictObject:
    """Converting BDD100K Detection Set to COCO format."""
    naming_replacement_dict = {
        "person": "pedestrian",
        "motor": "motorcycle",
        "bike": "bicycle",
    }

    coco: DictObject = defaultdict(list)
    coco["categories"] = [
        {"supercategory": "none", "id": 1, "name": "pedestrian"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "truck"},
        {"supercategory": "none", "id": 5, "name": "bus"},
        {"supercategory": "none", "id": 6, "name": "train"},
        {"supercategory": "none", "id": 7, "name": "motorcycle"},
        {"supercategory": "none", "id": 8, "name": "bicycle"},
        {"supercategory": "none", "id": 9, "name": "traffic light"},
        {"supercategory": "none", "id": 10, "name": "traffic sign"},
        {"supercategory": "none", "id": 11, "name": "ignored"},
    ]
    attr_id_dict: DictObject = {
        frame["name"]: frame["id"] for frame in coco["categories"]
    }

    counter = 0
    for frame in tqdm(labels):
        counter += 1
        image: DictObject = dict()
        image["file_name"] = frame["name"]
        image["height"] = 720
        image["width"] = 1280

        image["id"] = counter

        if frame["labels"]:
            for label in frame["labels"]:
                # skip for drivable area and lane marking
                if "box2d" not in label:
                    continue
                annotation: DictObject = dict()
                annotation["iscrowd"] = (
                    int(label["attributes"]["crowd"])
                    if "crowd" in label["attributes"]
                    else 0
                )
                annotation["image_id"] = image["id"]

                x1 = label["box2d"]["x1"]
                y1 = label["box2d"]["y1"]
                x2 = label["box2d"]["x2"]
                y2 = label["box2d"]["y2"]

                annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                annotation["area"] = float((x2 - x1) * (y2 - y1))
                # fix legacy naming
                if label["category"] in naming_replacement_dict:
                    label["category"] = naming_replacement_dict[
                        label["category"]
                    ]
                category_ignored = label["category"] not in attr_id_dict

                annotation["category_id"] = (
                    attr_id_dict["ignored"]
                    if category_ignored
                    else attr_id_dict[label["category"]]
                )
                annotation["ignore"] = int(category_ignored)
                annotation["id"] = label["id"]
                annotation["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                coco["annotations"].append(annotation)
        else:
            continue

        coco["images"].append(image)
    coco["type"] = "instances"

    return coco


def bdd100k2coco_track(labels: List[DictObject]) -> DictObject:
    """Converting BDD100K Tracking Set to COCO format."""
    coco = defaultdict(list)
    coco["categories"] = [
        {"supercategory": "none", "id": 1, "name": "pedestrian"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "truck"},
        {"supercategory": "none", "id": 5, "name": "bus"},
        {"supercategory": "none", "id": 6, "name": "train"},
        {"supercategory": "none", "id": 7, "name": "motorcycle"},
        {"supercategory": "none", "id": 8, "name": "bicycle"},
        {"supercategory": "none", "id": 9, "name": "ignored"},
    ]
    attr_id_dict = {i["name"]: i["id"] for i in coco["categories"]}

    video_id, image_id, ann_id, global_instance_id = 0, 0, 0, 0
    no_ann = 0

    for video_anns in tqdm(labels):
        instance_id_maps: DictObject = dict()

        # videos
        video = dict(id=video_id, name=video_anns[0]["video_name"])
        coco["videos"].append(video)

        # images
        for image_anns in video_anns:
            video_name = "-".join(image_anns["name"].split("-")[:-1])
            image = dict(
                file_name=os.path.join(video_name, image_anns["name"]),
                height=720,
                width=1280,
                id=image_id,
                video_id=video_id,
                index=image_anns["index"],
            )
            coco["images"].append(image)

            # annotations
            for lbl in image_anns["labels"]:
                category_ignored = False
                if lbl["category"] not in attr_id_dict.keys():
                    lbl["category"] = "ignored"
                    category_ignored = True

                bdd100k_id = lbl["id"]
                if bdd100k_id in instance_id_maps.keys():
                    instance_id = instance_id_maps[bdd100k_id]
                else:
                    instance_id = global_instance_id
                    global_instance_id += 1
                    instance_id_maps[bdd100k_id] = instance_id

                x1 = lbl["box2d"]["x1"]
                x2 = lbl["box2d"]["x2"]
                y1 = lbl["box2d"]["y1"]
                y2 = lbl["box2d"]["y2"]
                area = float((x2 - x1) * (y2 - y1))
                ann = dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=attr_id_dict[lbl["category"]],
                    instance_id=instance_id,
                    is_occluded=lbl["attributes"]["Occluded"],
                    is_truncated=lbl["attributes"]["Truncated"],
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    area=area,
                    iscrowd=int(lbl["attributes"]["Crowd"]),
                    ignore=int(category_ignored),
                    segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
                )

                coco["annotations"].append(ann)
                ann_id += 1
            if len(image_anns["labels"]) == 0:
                no_ann += 1

            image_id += 1
        video_id += 1

    return coco


def main() -> None:
    """Main function."""
    args = parse_arguments()

    print("Loading...")
    if os.path.isdir(args.in_path):
        # labels are provided in multiple json files in a folder
        labels = []
        for p in sorted(os.listdir(args.in_path)):
            with open(os.path.join(args.in_path, p)) as f:
                labels.append(json.load(f))
    else:
        with open(args.in_path) as f:
            labels = json.load(f)

    print("Converting...")
    out_fn = os.path.join(args.out_path)

    if args.mode == "det":
        coco = bdd100k2coco_det(labels)
    elif args.mode == "track":
        coco = bdd100k2coco_track(labels)

    print("Saving...")
    with open(out_fn, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    main()
