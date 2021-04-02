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
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util  # type: ignore
from scalabel.label.typing import Frame, Label
from skimage import measure
from tqdm import tqdm

from ..common.logger import logger
from ..common.typing import DictAny
from ..common.utils import IGNORE_MAP, NAME_MAPPING, group_and_sort, init, read


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
        "-mm",
        "--mask-mode",
        default="rle",
        choices=["rle", "polygon"],
        help="conversion mode: rle or polygon.",
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
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for mot evaluation",
    )
    return parser


def close_contour(contour: np.ndarray) -> np.ndarray:
    """Explicitly close the contour."""
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def mask_to_polygon(
    binary_mask: np.ndarray, x_1: int, y_1: int, tolerance: int = 2
) -> List[List[float]]:
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
    image: DictAny, image_name: str, image_id: int, video_name: str = ""
) -> None:
    """Set attributes for the image dict."""
    image.update(
        dict(
            file_name=os.path.join(video_name, image_name),
            height=720,
            width=1280,
            id=image_id,
        )
    )


def set_object_attributes(
    annotation: DictAny, label: Label, ignore: bool
) -> None:
    """Set attributes for the ann dict."""
    attributes = label.attributes
    if attributes is None:
        return
    iscrowd = bool(attributes.get("crowd", False))
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


def set_box_object_geometry(annotation: DictAny, label: Label) -> None:
    """Parsing bbox, area, polygon for bbox ann."""
    box_2d = label.box_2d
    if box_2d is None:
        return
    x1 = box_2d.x1
    y1 = box_2d.y1
    x2 = box_2d.x2
    y2 = box_2d.y2

    annotation.update(
        dict(
            bbox=[x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            area=float((x2 - x1 + 1) * (y2 - y1 + 1)),
            segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
        )
    )


def set_seg_object_geometry(
    ann: DictAny, mask: np.ndarray, mask_mode: str = "rle"
) -> None:
    """Parsing bbox, area, polygon from seg ann."""
    if not mask.sum():
        return

    if mask_mode == "polygon":
        x_inds = np.nonzero(np.sum(mask, axis=0))[0]
        y_inds = np.nonzero(np.sum(mask, axis=1))[0]
        x1, x2 = np.min(x_inds), np.max(x_inds)
        y1, y2 = np.min(y_inds), np.max(y_inds)
        mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        segmentation = mask_to_polygon(mask, x1, y1)
        bbox = np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1]).tolist()
        area = np.sum(mask).tolist()
    elif mask_mode == "rle":
        segmentation = mask_util.encode(
            np.array(mask[:, :, None], order="F", dtype="uint8")
        )[0]
        segmentation["counts"] = segmentation["counts"].decode(  # type: ignore
            "utf-8"
        )
        bbox = mask_util.toBbox(segmentation).tolist()
        area = mask_util.area(segmentation).tolist()

    ann.update(dict(bbox=bbox, area=area, segmentation=segmentation))


def bdd100k2coco_det(
    labels: List[Frame],
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> DictAny:
    """Converting BDD100K Detection Set to COCO format."""
    coco, cat_name2id = init(mode="det", ignore_as_class=ignore_as_class)
    coco["type"] = "instances"
    image_id, ann_id = 1, 1

    for frame in tqdm(labels):
        image: DictAny = dict()
        set_image_attributes(image, frame.name, image_id)
        coco["images"].append(image)

        if frame.labels is None:
            continue
        for label in frame.labels:
            if label.box_2d is None:
                continue

            category_ignored, category_id = process_category(
                label.category, ignore_as_class, cat_name2id
            )
            if remove_ignore and category_ignored:
                continue

            annotation = dict(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                bdd100k_id=str(label.id),
            )
            set_object_attributes(annotation, label, category_ignored)
            set_box_object_geometry(annotation, label)
            coco["annotations"].append(annotation)

            ann_id += 1
        image_id += 1

    return coco


def bdd100k2coco_box_track(
    labels: List[List[Frame]],
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
        video_name = video_anns[0].video_name
        video = dict(id=video_id, name=video_name)
        coco["videos"].append(video)

        # images
        for image_anns in video_anns:
            image = dict(video_id=video_id, frame_id=image_anns.index)
            image_name = os.path.join(video_name, image_anns.name)
            set_image_attributes(image, image_name, image_id, video_name)
            coco["images"].append(image)

            # annotations
            for label in image_anns.labels:
                if label.box_2d is None:
                    continue

                category_ignored, category_id = process_category(
                    label.category, ignore_as_class, cat_name2id
                )
                if remove_ignore and category_ignored:
                    continue

                bdd100k_id = str(label.id)
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
                set_object_attributes(ann, label, category_ignored)
                set_box_object_geometry(ann, label)
                coco["annotations"].append(ann)

                ann_id += 1
            image_id += 1
        video_id += 1

    return coco


def bitmask2coco(
    annotations: List[DictAny],
    mask_name: str,
    category_ids: List[int],
    instance_ids: List[int],
    mask_mode: str = "rle",
) -> List[DictAny]:
    """Convert bitmasks annotations of an image to RLEs or polygons."""
    bitmask = np.asarray(Image.open(mask_name)).astype(np.int32)
    category_map = bitmask[..., 0]
    instance_map = (bitmask[..., 2] << 2) + bitmask[..., 3]
    for annotation, category_id, instance_id in zip(
        annotations, category_ids, instance_ids
    ):
        mask = np.logical_and(
            category_map == category_id, instance_map == instance_id
        )
        set_seg_object_geometry(annotation, mask, mask_mode)
    annotations = [
        ann for ann in annotations if "bbox" in ann and "segmentation" in ann
    ]
    return annotations


def coco_parellel_conversion(
    annotations_list: List[List[DictAny]],
    mask_names: List[str],
    category_ids_list: List[List[int]],
    instance_ids_list: List[List[int]],
    mask_mode: str = "rle",
    nproc: int = 4,
) -> List[List[DictAny]]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    pool = Pool(nproc)
    annotations_list = pool.starmap(
        bitmask2coco,
        tqdm(
            zip(
                annotations_list,
                mask_names,
                category_ids_list,
                instance_ids_list,
                [mask_mode for _ in range(len(annotations_list))],
            ),
            total=len(annotations_list),
        ),
    )
    pool.close()
    return annotations_list


def bdd100k2coco_ins_seg(
    labels: List[Frame],
    mask_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> DictAny:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    coco, cat_name2id = init(mode="track", ignore_as_class=ignore_as_class)
    coco["type"] = "instances"
    image_id, ann_id = 1, 1

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[DictAny]] = []

    logger.info("Collecting bitmasks...")

    for frame in tqdm(labels):
        instance_id = 1
        image: DictAny = dict()
        set_image_attributes(image, frame.name, image_id)
        coco["images"].append(image)

        mask_name = os.path.join(
            mask_base,
            frame["name"].replace(".jpg", ".png"),
        )
        mask_names.append(mask_name)

        category_ids: List[int] = []
        instance_ids: List[int] = []
        annotations: List[DictAny] = []

        # annotations
        for label in frame.labels:
            if label.poly_2d is None:
                continue

            category_ignored, category_id = process_category(
                label.category, ignore_as_class, cat_name2id
            )
            if category_ignored and remove_ignore:
                continue

            annotation: DictAny = dict(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                bdd100k_id=str(label.id),
            )
            set_object_attributes(annotation, label, category_ignored)

            category_ids.append(category_id)
            instance_ids.append(instance_id)
            annotations.append(annotation)
            ann_id += 1
            instance_id += 1

        category_ids_list.append(category_ids)
        instance_ids_list.append(instance_ids)
        annotations_list.append(annotations)
        image_id += 2

    annotations_list = coco_parellel_conversion(
        annotations_list,
        mask_names,
        category_ids_list,
        instance_ids_list,
        mask_mode,
        nproc,
    )
    for annotations in annotations_list:
        coco["annotations"].extend(annotations)

    return coco


def bdd100k2coco_seg_track(
    labels: List[List[Frame]],
    mask_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> DictAny:
    """Converting BDD100K Segmentation Tracking Set to COCO format."""
    coco, cat_name2id = init(mode="track", ignore_as_class=ignore_as_class)
    video_id, image_id, ann_id = 1, 1, 1

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[DictAny]] = []

    for video_anns in tqdm(labels):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        # videos
        video_name = video_anns[0].video_name
        video = dict(id=video_id, name=video_name)
        coco["videos"].append(video)

        # images
        for image_anns in video_anns:
            image = dict(video_id=video_id, frame_id=image_anns.index)
            image_name = os.path.join(video_name, image_anns.name)
            set_image_attributes(image, image_name, image_id, video_name)
            coco["images"].append(image)

            mask_name = os.path.join(
                mask_base,
                video_name,
                image_anns.name.replace(".jpg", ".png"),
            )
            mask_names.append(mask_name)

            category_ids: List[int] = []
            instance_ids: List[int] = []
            annotations: List[DictAny] = []

            # annotations
            for label in image_anns.labels:
                if label.poly_2d is None:
                    continue

                category_ignored, category_id = process_category(
                    label.category, ignore_as_class, cat_name2id
                )
                if category_ignored and remove_ignore:
                    continue

                bdd100k_id = str(label.id)
                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, bdd100k_id
                )

                annotation = dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    bdd100k_id=bdd100k_id,
                )
                set_object_attributes(annotation, label, category_ignored)

                category_ids.append(category_id)
                instance_ids.append(instance_id)
                annotations.append(annotation)
                ann_id += 1

            category_ids_list.append(category_ids)
            instance_ids_list.append(instance_ids)
            annotations_list.append(annotations)
            image_id += 1

        video_id += 1

    annotations_list = coco_parellel_conversion(
        annotations_list,
        mask_names,
        category_ids_list,
        instance_ids_list,
        mask_mode,
        nproc,
    )
    for annotations in annotations_list:
        coco["annotations"].extend(annotations)

    return coco


def start_converting(
    parser_def_func: Callable[[], argparse.ArgumentParser]
) -> Tuple[argparse.Namespace, List[Frame]]:
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

    return args, labels


def main() -> None:
    """Main function."""
    args, labels = start_converting(parser_definition_coco)

    if args.mode == "det":
        coco = bdd100k2coco_det(
            labels, args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "ins_seg":
        coco = bdd100k2coco_ins_seg(
            labels,
            args.mask_base,
            args.ignore_as_class,
            args.remove_ignore,
            args.mask_mode,
            args.nproc,
        )
    elif args.mode == "box_track":
        coco = bdd100k2coco_box_track(
            group_and_sort(labels), args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "seg_track":
        coco = bdd100k2coco_seg_track(
            group_and_sort(labels),
            args.mask_base,
            args.ignore_as_class,
            args.remove_ignore,
            args.mask_mode,
            args.nproc,
        )

    logger.info("Saving converted annotations to disk...")
    with open(args.out_path, "w") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
