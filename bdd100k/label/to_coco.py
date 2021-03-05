"""Convert BDD100K to COCO format.

The annotation files in BDD100K format has additional annotaitons
('other person', 'other vehicle' and 'trail') besides the considered
categories ('car', 'pedestrian', 'truck', etc.) to indicate the uncertain
regions. Given the different handlings of these additional classes, we
provide three options to process the labels when converting them into COCO
format.
1. Ignore the labels. This is the default setting and is often used for
evaluation. CocoAPIs have native support for ignored annotations.
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L370
2. Remove the annotations from the label file. By adding the
flag `--remove-ignore`, the script will remove all the ignored annotations.
3. Use `ignore` as a separate class and the user can decide how to utilize
the annotations in `ignored` class. To achieve this, add the flag
`--ignore-as-class`.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm

from ..common.logger import logger
from ..common.typing import DictAny, ListAny


def parse_arguments() -> argparse.Namespace:
    """arguments."""
    parser = argparse.ArgumentParser(description="BDD100K to COCO format")
    parser.add_argument(
        "-i",
        "--in-path",
        default="/input/path/",
        help="Path to detection JSON file or tracking base folder.",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        default="/output/path",
        help="Path to save coco formatted label file.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["det", "track", "seg_track"],
        help="conversion mode: detection or tracking.",
    )
    parser.add_argument(
        "-mb",
        "--mask-base",
        help="Path to the BitMasks base folder.",
    )
    parser.add_argument(
        "-ri",
        "--remove-ignore",
        action="store_true",
        help="Remove the ignored annotations from the label file.",
    )
    parser.add_argument(
        "-ic",
        "--ignore-as-class",
        action="store_true",
        help="Put the ignored annotations to the `ignored` category.",
    )
    return parser.parse_args()


def init(
    mode: str = "det", ignore_as_class: bool = False
) -> Tuple[DictAny, Dict[str, str], DictAny]:
    """Initialze the annotation dictionary."""
    coco: DictAny = defaultdict(list)
    coco["categories"] = [
        {"supercategory": "human", "id": 1, "name": "pedestrian"},
        {"supercategory": "human", "id": 2, "name": "rider"},
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        {"supercategory": "vehicle", "id": 4, "name": "truck"},
        {"supercategory": "vehicle", "id": 5, "name": "bus"},
        {"supercategory": "vehicle", "id": 6, "name": "train"},
        {"supercategory": "bike", "id": 7, "name": "motorcycle"},
        {"supercategory": "bike", "id": 8, "name": "bicycle"},
    ]
    if mode == "det":
        coco["categories"] += [
            {
                "supercategory": "traffic light",
                "id": 9,
                "name": "traffic light",
            },
            {
                "supercategory": "traffic sign",
                "id": 10,
                "name": "traffic sign",
            },
        ]

    if ignore_as_class:
        coco["categories"].append(
            {
                "supercategory": "none",
                "id": len(coco["categories"]) + 1,
                "name": "ignored",
            }
        )

    # Mapping the ignored classes to standard classes.
    ignore_map = {
        "other person": "pedestrian",
        "other vehicle": "car",
        "trailer": "truck",
    }

    attr_id_dict: DictAny = {
        frame["name"]: frame["id"] for frame in coco["categories"]
    }
    return coco, ignore_map, attr_id_dict


def bdd100k2coco_det(
    labels: List[DictAny],
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Detection Set to COCO format."""
    naming_replacement_dict = {
        "person": "pedestrian",
        "motor": "motorcycle",
        "bike": "bicycle",
    }

    coco, ignore_map, attr_id_dict = init(
        mode="det", ignore_as_class=ignore_as_class
    )
    counter = 0
    label_counter = 0
    for frame in tqdm(labels):
        counter += 1
        image: DictAny = dict()
        image["file_name"] = frame["name"]
        image["height"] = 720
        image["width"] = 1280

        image["id"] = counter

        if frame["labels"]:
            for label in frame["labels"]:
                # skip for drivable area and lane marking
                if "box2d" not in label:
                    continue
                label_counter += 1
                annotation: DictAny = dict()
                annotation["iscrowd"] = (
                    int(label["attributes"]["Crowd"])
                    if "Crowd" in label["attributes"]
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

                if remove_ignore and category_ignored:
                    continue

                # Merging the ignored examples to car but
                # the annotation is ignored for training and evaluation.
                if category_ignored:
                    if ignore_as_class:
                        cls_id = attr_id_dict["ignored"]
                    else:
                        cls_id = attr_id_dict[ignore_map[label["category"]]]
                else:
                    cls_id = attr_id_dict[label["category"]]
                annotation["category_id"] = cls_id
                if ignore_as_class:
                    annotation["ignore"] = 0
                else:
                    annotation["ignore"] = int(category_ignored)
                # COCOAPIs only ignores the crowd region.
                annotation["iscrowd"] = annotation["iscrowd"] or int(
                    category_ignored
                )

                annotation["id"] = label_counter
                # save the original bdd100k_id for backup.
                # The BDD100K ID might be string in the future.
                annotation["bdd100k_id"] = label["id"]
                annotation["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                coco["annotations"].append(annotation)
        else:
            continue

        coco["images"].append(image)
    coco["type"] = "instances"

    return coco


def bdd100k2coco_track(
    labels: List[DictAny],
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Tracking Set to COCO format."""
    coco, ignore_map, attr_id_dict = init(
        mode="track", ignore_as_class=ignore_as_class
    )

    video_id, image_id, ann_id = 1, 1, 1
    no_ann = 0

    for video_anns in tqdm(labels):
        global_instance_id: int = 1
        instance_id_maps: DictAny = dict()

        # videos
        video = dict(id=video_id, name=video_anns[0]["video_name"])
        coco["videos"].append(video)
        video_name = video_anns[0]["video_name"]

        # images
        for image_anns in video_anns:
            image = dict(
                file_name=os.path.join(video_name, image_anns["name"]),
                height=720,
                width=1280,
                id=image_id,
                video_id=video_id,
                frame_id=image_anns["index"],
            )
            coco["images"].append(image)

            # annotations
            for label in image_anns["labels"]:
                category_ignored = False
                if label["category"] not in attr_id_dict.keys():
                    if ignore_as_class:
                        label["category"] = "ignored"
                        category_ignored = False
                    else:
                        label["category"] = ignore_map[label["category"]]
                        category_ignored = True
                    if category_ignored and remove_ignore:
                        # remove the ignored annotations
                        continue

                bdd100k_id = label["id"]
                if bdd100k_id in instance_id_maps.keys():
                    instance_id = instance_id_maps[bdd100k_id]
                else:
                    instance_id = global_instance_id
                    global_instance_id += 1
                    instance_id_maps[bdd100k_id] = instance_id

                x1 = label["box2d"]["x1"]
                x2 = label["box2d"]["x2"]
                y1 = label["box2d"]["y1"]
                y2 = label["box2d"]["y2"]
                area = float((x2 - x1) * (y2 - y1))
                ann = dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=attr_id_dict[label["category"]],
                    instance_id=instance_id,
                    bdd100k_id=bdd100k_id,
                    occluded=label["attributes"]["Occluded"],
                    truncated=label["attributes"]["Truncated"],
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    area=area,
                    iscrowd=int(label["attributes"]["Crowd"])
                    or int(category_ignored),
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


def close_contour(contour: ListAny) -> ListAny:
    """Explicitly close the contour."""
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def mask_to_polygon(
    binary_mask: np.array, tolerance: int = 2
) -> List[ListAny]:
    """Convert BitMask to polygon."""
    polygons = []
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def bdd100k2coco_seg_track(
    labels: List[DictAny],
    mask_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Tracking Set to COCO format."""
    coco, ignore_map, attr_id_dict = init(
        mode="track", ignore_as_class=ignore_as_class
    )

    video_id, image_id, ann_id = 1, 1, 1
    no_ann = 0

    for video_anns in tqdm(labels):
        global_instance_id: int = 1
        instance_id_maps: DictAny = dict()

        # videos
        video = dict(id=video_id, name=video_anns[0]["video_name"])
        coco["videos"].append(video)
        video_name = video_anns[0]["video_name"]

        # images
        for image_anns in video_anns:
            image_name = os.path.split(image_anns["name"])[-1]
            image = dict(
                file_name=os.path.join(video_name, image_name),
                height=720,
                width=1280,
                id=image_id,
                video_id=video_id,
                frame_id=image_anns["index"],
            )
            coco["images"].append(image)

            mask_name = os.path.join(
                mask_base,
                video_name,
                image_anns["name"].replace(".jpg", ".png"),
            )
            bitmask = np.asarray(Image.open(mask_name)).astype(np.int32)
            category_map = bitmask[:, :, 0]
            instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

            # annotations
            for label in image_anns["labels"]:
                if "poly2d" not in label:
                    continue

                category_ignored = False
                # fix legacy naming
                if label["category"] not in attr_id_dict.keys():
                    if ignore_as_class:
                        label["category"] = "ignored"
                        category_ignored = False
                    else:
                        label["category"] = ignore_map[label["category"]]
                        category_ignored = True
                    if category_ignored and remove_ignore:
                        # remove the ignored annotations
                        continue
                category_id = attr_id_dict[label["category"]]

                bdd100k_id = label["id"]
                if bdd100k_id in instance_id_maps.keys():
                    instance_id = instance_id_maps[bdd100k_id]
                else:
                    instance_id = global_instance_id
                    global_instance_id += 1
                    instance_id_maps[bdd100k_id] = instance_id

                mask = np.logical_and(
                    category_map == category_id,
                    instance_map == instance_id,
                )
                if not mask.sum():
                    continue

                x_inds = np.sum(mask, axis=0).nonzero()[0]
                y_inds = np.sum(mask, axis=1).nonzero()[0]
                x1, x2 = np.min(x_inds), np.max(x_inds) + 1
                y1, y2 = np.min(y_inds), np.max(y_inds) + 1

                mask = mask[y1:y2, x1:x2]
                area = np.sum(mask).tolist()
                polygon = mask_to_polygon(mask)
                for p in polygon:
                    for n, _ in enumerate(p):
                        if n % 2 == 0:
                            p[n] = (p[n] + x1).tolist()
                        else:
                            p[n] = (p[n] + y1).tolist()

                ann = dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    bdd100k_id=bdd100k_id,
                    occluded=label["attributes"]["Occluded"],
                    truncated=label["attributes"]["Truncated"],
                    bbox=np.array([x1, y1, x2 - x1, y2 - y1]).tolist(),
                    area=area,
                    iscrowd=int(label["attributes"]["Crowd"])
                    or int(category_ignored),
                    ignore=int(category_ignored),
                    segmentation=polygon,
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

    logger.info(
        "Mode: %s\nremove-ignore: %s\nignore-as-class: %s",
        args.mode,
        args.remove_ignore,
        args.ignore_as_class,
    )
    logger.info("Loading annotations...")
    if os.path.isdir(args.in_path):
        # labels are provided in multiple json files in a folder
        labels = []
        for p in sorted(os.listdir(args.in_path)):
            with open(os.path.join(args.in_path, p)) as f:
                labels.append(json.load(f))
    else:
        with open(args.in_path) as f:
            labels = json.load(f)

    logger.info("Converting annotations...")
    out_fn = os.path.join(args.out_path)

    if args.mode == "det":
        coco = bdd100k2coco_det(
            labels, args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "track":
        coco = bdd100k2coco_track(
            labels, args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "seg_track":
        coco = bdd100k2coco_seg_track(
            labels, args.mask_base, args.ignore_as_class, args.remove_ignore
        )

    logger.info("Saving converted annotations to disk...")
    with open(out_fn, "w") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
