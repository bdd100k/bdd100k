"""Convert BDD100K to COCO format.

The annotation files in BDD100K format has additional annotations
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
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm

from ..common.logger import logger
from ..common.typing import DictAny, ListAny
from ..common.utils import IGNORE_MAP, NAME_MAPPING, init, read


def parser_definition_coco() -> argparse.ArgumentParser:
    """Definition of the parser."""
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
        default="/output/path/out.json",
        help="Path to save coco formatted label file.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["det", "ins_seg", "box_track", "seg_track"],
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
    return parser


def close_contour(contour: ListAny) -> ListAny:
    """Explicitly close the contour."""
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def mask_to_polygon(
    binary_mask: np.array, x_1: int, y_1: int, tolerance: int = 2
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
        for i, _ in enumerate(segmentation):
            if i % 2 == 0:
                segmentation[i] = (segmentation[i] + x_1).tolist()
            else:
                segmentation[i] = (segmentation[i] + y_1).tolist()

        polygons.append(segmentation)

    return polygons


def set_image_attributes(
    image: DictAny, image_name: str, image_id: int
) -> None:
    """Set attributes for the image dict."""
    image.update(
        dict(
            file_name=image_name,
            height=720,
            width=1280,
            id=image_id,
        )
    )


def set_object_attributes(annotation: DictAny, label: DictAny) -> None:
    """Set attributes for the ann dict."""
    iscrowd = bool(label["attributes"].get("Crowd", False))
    ignore = bool(label["category_ignored"])
    annotation.update(
        dict(
            iscrowd=int(iscrowd or ignore),
            ignore=int(ignore),
        )
    )


def process_category(
    category_name: str, ignore_as_class: bool, cat_name2id: Dict[str, int]
) -> Tuple[bool, int]:
    """Check whether the category should be ignored and get its ID."""
    category_name = NAME_MAPPING.get(category_name, category_name)
    if category_name not in cat_name2id:
        if ignore_as_class:
            category_name = "ignored"
            category_ignored = False
        else:
            category_name = IGNORE_MAP[category_name]
            category_ignored = True
    else:
        category_ignored = False
    category_id = cat_name2id[category_name]
    return category_ignored, category_id


def get_instance_id(
    instance_id_maps: Dict[str, int], global_instance_id: int, bdd100k_id: str
) -> Tuple[int, int]:
    """Get instance id given its corresponding bdd100k id."""
    if bdd100k_id in instance_id_maps.keys():
        instance_id = instance_id_maps[bdd100k_id]
    else:
        instance_id = global_instance_id
        global_instance_id += 1
        instance_id_maps[bdd100k_id] = instance_id
    return instance_id, global_instance_id


def set_box_object_geometry(annotation: DictAny, label: DictAny) -> None:
    """Parsing bbox, area, polygon for bbox ann."""
    x1 = label["box2d"]["x1"]
    y1 = label["box2d"]["y1"]
    x2 = label["box2d"]["x2"]
    y2 = label["box2d"]["y2"]

    annotation.update(
        dict(
            bbox=[x1, y1, x2 - x1, y2 - y1],
            area=float((x2 - x1) * (y2 - y1)),
            segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
        )
    )


def set_seg_object_geometry(ann: DictAny, mask: np.ndarray) -> None:
    """Parsing bbox, area, polygon from seg ann."""
    if not mask.sum():
        return

    x_inds = np.sum(mask, axis=0).nonzero()[0]
    y_inds = np.sum(mask, axis=1).nonzero()[0]
    x1, x2 = np.min(x_inds), np.max(x_inds) + 1
    y1, y2 = np.min(y_inds), np.max(y_inds) + 1
    mask = mask[y1:y2, x1:x2]

    ann.update(
        dict(
            bbox=np.array([x1, y1, x2 - x1, y2 - y1]).tolist(),
            area=np.sum(mask).tolist(),
            segmentation=mask_to_polygon(mask, x1, y1),
        )
    )


def bdd100k2coco_det(
    labels: List[List[DictAny]],
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Detection Set to COCO format."""
    assert len(labels) == 1

    coco, cat_name2id = init(mode="det", ignore_as_class=ignore_as_class)
    coco["type"] = "instances"
    image_id, ann_id = 1, 1

    for frame in tqdm(labels[0]):
        image: DictAny = dict()
        set_image_attributes(image, frame["name"], image_id)
        coco["images"].append(image)

        if not frame["labels"]:
            continue
        for label in frame["labels"]:
            if "box2d" not in label:
                continue

            category_ignored, category_id = process_category(
                label["category"], ignore_as_class, cat_name2id
            )
            if remove_ignore and category_ignored:
                continue
            label["category_ignored"] = category_ignored

            annotation = dict(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                bdd100k_id=str(label["id"]),
            )
            set_object_attributes(annotation, label)
            set_box_object_geometry(annotation, label)
            coco["annotations"].append(annotation)

            ann_id += 1
        image_id += 1

    return coco


def bdd100k2coco_box_track(
    labels: List[List[DictAny]],
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Box Tracking Set to COCO format."""
    coco, cat_name2id = init(mode="track", ignore_as_class=ignore_as_class)
    video_id, image_id, ann_id = 1, 1, 1

    for video_anns in tqdm(labels):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        # videos
        video_name = video_anns[0]["video_name"]
        video = dict(id=video_id, name=video_name)
        coco["videos"].append(video)

        # images
        for image_anns in video_anns:
            image = dict(video_id=video_id, frame_id=image_anns["index"])
            image_name = os.path.join(video_name, image_anns["name"])
            set_image_attributes(image, image_name, image_id)
            coco["images"].append(image)

            # annotations
            for label in image_anns["labels"]:
                if "box2d" not in label:
                    continue

                category_ignored, category_id = process_category(
                    label["category"], ignore_as_class, cat_name2id
                )
                if remove_ignore and category_ignored:
                    continue
                label["category_ignored"] = category_ignored

                bdd100k_id = str(label["id"])
                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, bdd100k_id
                )
                ann = dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    bdd100k_id=bdd100k_id,
                )
                set_object_attributes(ann, label)
                set_box_object_geometry(ann, label)
                coco["annotations"].append(ann)

                ann_id += 1
            image_id += 1
        video_id += 1

    return coco


def bdd100k2coco_seg_track(
    labels: List[List[DictAny]],
    mask_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Segmentation Tracking Set to COCO format."""
    coco, cat_name2id = init(mode="track", ignore_as_class=ignore_as_class)
    video_id, image_id, ann_id = 1, 1, 1

    for video_anns in tqdm(labels):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        # videos
        video_name = video_anns[0]["video_name"]
        video = dict(id=video_id, name=video_name)
        coco["videos"].append(video)

        # images
        for image_anns in video_anns:
            image = dict(video_id=video_id, frame_id=image_anns["index"])
            image_name = os.path.join(video_name, image_anns["name"])
            set_image_attributes(image, image_name, image_id)
            coco["images"].append(image)

            mask_name = os.path.join(
                mask_base,
                video_name,
                image_anns["name"].replace(".jpg", ".png"),
            )
            bitmask = np.asarray(Image.open(mask_name)).astype(np.int32)
            category_map = bitmask[..., 0]
            instance_map = (bitmask[..., 2] << 2) + bitmask[..., 3]

            # annotations
            for label in image_anns["labels"]:
                if "poly2d" not in label:
                    continue

                category_ignored, category_id = process_category(
                    label["category"], ignore_as_class, cat_name2id
                )
                label["category_ignored"] = category_ignored
                if category_ignored and remove_ignore:
                    continue

                bdd100k_id = str(label["id"])
                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, bdd100k_id
                )

                mask = np.logical_and(
                    category_map == category_id, instance_map == instance_id
                )

                ann = dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    bdd100k_id=bdd100k_id,
                )
                set_object_attributes(ann, label)
                set_seg_object_geometry(ann, mask)
                coco["annotations"].append(ann)

                ann_id += 1
            image_id += 1
        video_id += 1

    return coco


def start_converting(
    parser_def_func: Callable[[], argparse.ArgumentParser]
) -> Tuple[argparse.Namespace, List[List[DictAny]]]:
    """Parses arguments, and logs settings."""
    parser = parser_def_func()
    args = parser.parse_args()

    logger.info(
        "Mode: %s\nremove-ignore: %s\nignore-as-class: %s",
        args.mode,
        args.remove_ignore,
        args.ignore_as_class,
    )
    logger.info("Loading annotations...")
    labels = read(args.in_path)

    logger.info("Converting annotations...")
    return args, labels


def main() -> None:
    """Main function."""
    args, labels = start_converting(parser_definition_coco)

    if args.mode == "det":
        coco = bdd100k2coco_det(
            labels, args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "box_track":
        coco = bdd100k2coco_box_track(
            labels, args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "seg_track":
        coco = bdd100k2coco_seg_track(
            labels, args.mask_base, args.ignore_as_class, args.remove_ignore
        )

    logger.info("Saving converted annotations to disk...")
    with open(args.out_path, "w") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
