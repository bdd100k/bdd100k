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
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scalabel.label.coco_typing import (
    AnnType,
    CatType,
    GtType,
    ImgType,
    VidType,
)
from scalabel.label.to_coco import (
    get_instance_id,
    get_object_attributes,
    group_and_sort,
    parse_arguments,
    process_category,
    scalabel2coco_box_track,
    scalabel2coco_detection,
    set_seg_object_geometry,
)
from scalabel.label.typing import Frame
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import load_categories, read


def parser_arguments_to_coco() -> argparse.Namespace:
    """Definition of the parser."""
    parser = argparse.ArgumentParser(description="BDD100K to COCO format")
    parser.add_argument(
        "-mb",
        "--mask-base",
        help="Path to the BitMasks base folder.",
    )
    return parser.parse_args()


def bitmask2coco(
    annotations: List[AnnType],
    mask_name: str,
    category_ids: List[int],
    instance_ids: List[int],
    mask_mode: str = "rle",
) -> List[AnnType]:
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
        annotation = set_seg_object_geometry(annotation, mask, mask_mode)
    annotations = [
        ann for ann in annotations if "bbox" in ann and "segmentation" in ann
    ]
    return annotations


def coco_parellel_conversion(
    annotations_list: List[List[AnnType]],
    mask_names: List[str],
    category_ids_list: List[List[int]],
    instance_ids_list: List[List[int]],
    mask_mode: str = "rle",
    nproc: int = 4,
) -> List[List[AnnType]]:
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

    return annotations_list


def bdd100k2coco_ins_seg(
    mask_base: str,
    shape: Tuple[int, int],
    frames: List[Frame],
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    images = []
    image_id, ann_id = 1, 1

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[AnnType]] = []

    logger.info("Collecting bitmasks...")

    for frame in tqdm(frames):
        image = ImgType(
            id=image_id,
            file_name=frame.name,
            height=shape[0],
            width=shape[1],
        )
        images.append(image)
        instance_id = 1

        mask_name = os.path.join(
            mask_base,
            frame["name"].replace(".jpg", ".png"),
        )
        mask_names.append(mask_name)

        category_ids: List[int] = []
        instance_ids: List[int] = []
        annotations: List[AnnType] = []

        for label in frame.labels:
            if label.poly_2d is None:
                continue
            category_ignored, category_id = process_category(
                label.category,
                categories,
                name_mapping,
                ignore_mapping,
                ignore_as_class=ignore_as_class,
            )
            if category_ignored and remove_ignore:
                continue

            iscrowd, ignore = get_object_attributes(label, category_ignored)
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=str(label.id),
                iscrowd=iscrowd,
                ignore=ignore,
            )

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
    final_annotations: List[AnnType] = []
    for annotations in annotations_list:
        final_annotations.extend(annotations)

    return GtType(
        type="instances",
        categories=categories,
        images=images,
        annotations=final_annotations,
    )


def bdd100k2coco_seg_track(
    mask_base: str,
    shape: Tuple[int, int],
    frames: List[Frame],
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Converting BDD100K Segmentation Tracking Set to COCO format."""
    frames_list = group_and_sort(frames)
    videos: List[VidType] = []
    images: List[ImgType] = []
    video_id, image_id, ann_id = 1, 1, 1

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[AnnType]] = []

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_name = video_anns[0].video_name
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in video_anns:
            image = ImgType(
                video_id=video_id,
                frame_id=image_anns.frame_index,
                id=image_id,
                file_name=os.path.join(video_name, image_anns.name),
                height=shape[0],
                width=shape[1],
            )
            images.append(image)

            mask_name = os.path.join(
                mask_base,
                video_name,
                image_anns.name.replace(".jpg", ".png"),
            )
            mask_names.append(mask_name)

            category_ids: List[int] = []
            instance_ids: List[int] = []
            annotations: List[AnnType] = []

            for label in image_anns.labels:
                if label.poly_2d is None:
                    continue
                category_ignored, category_id = process_category(
                    label.category,
                    categories,
                    name_mapping,
                    ignore_mapping,
                    ignore_as_class=ignore_as_class,
                )
                if remove_ignore and category_ignored:
                    continue

                scalabel_id = str(label.id)
                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, scalabel_id
                )

                iscrowd, ignore = get_object_attributes(
                    label, category_ignored
                )
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=scalabel_id,
                    iscrowd=iscrowd,
                    ignore=ignore,
                )

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
    final_annotations: List[AnnType] = []
    for annotations in annotations_list:
        final_annotations.extend(annotations)

    return GtType(
        type="instances",
        categories=categories,
        videos=videos,
        images=images,
        annotations=final_annotations,
    )


def start_converting() -> Tuple[argparse.Namespace, List[Frame]]:
    """Parses arguments, and logs settings."""
    args = parse_arguments()
    logger.info(
        "Mode: %s\nremove-ignore: %s\nignore-as-class: %s",
        args.mode,
        args.remove_ignore,
        args.ignore_as_class,
    )
    logger.info("Loading annotations...")
    labels = read(args.label)
    logger.info("Start format converting...")

    return args, labels


def main() -> None:
    """Main function."""
    args, frames = start_converting()
    args_to_coco = parser_arguments_to_coco()
    categories, name_mapping, ignore_mapping = load_categories(
        mode=args.mode, ignore_as_class=args.ignore_as_class
    )

    if args.mode in ["det", "box_track"]:
        convert_func = dict(
            det=scalabel2coco_detection,
            box_track=scalabel2coco_box_track,
        )[args.mode]
    else:
        convert_func = partial(
            dict(
                ins_seg=bdd100k2coco_ins_seg,
                seg_track=bdd100k2coco_seg_track,
            )[args.mode],
            mask_base=args_to_coco.mask_base,
            mask_mode=args.mask_mode,
            nproc=args.nproc,
        )
    shape = (args.height, args.width)
    coco = convert_func(
        shape,
        frames,
        categories,
        name_mapping,
        ignore_mapping,
        args.ignore_as_class,
        args.remove_ignore,
    )

    logger.info("Saving converted annotations to disk...")
    with open(args.output, "w") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
