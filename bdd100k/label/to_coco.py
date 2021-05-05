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
from scalabel.label.io import group_and_sort, load
from scalabel.label.to_coco import (
    load_coco_config,
    process_category,
    scalabel2coco_box_track,
    scalabel2coco_detection,
    set_seg_object_geometry,
)
from scalabel.label.transforms import mask_to_bbox
from scalabel.label.typing import Frame
from tqdm import tqdm

from ..common.logger import logger
from ..common.typing import InstanceType
from ..common.utils import (
    DEFAULT_COCO_CONFIG,
    get_bdd100k_instance_id,
    get_bdd100k_object_attributes,
    group_and_sort_files,
    list_files,
)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="bdd100k to coco format")
    parser.add_argument(
        "-l",
        "--label",
        help=(
            "root directory of bdd100k label Json files or path to a label "
            "json file"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco formatted label file",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Height of images",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=[
            "det",
            "sem_seg",
            "drivable",
            "lane_mark",
            "ins_seg",
            "box_track",
            "seg_track",
        ],
        help="conversion mode.",
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
        "-mm",
        "--mask-mode",
        default="rle",
        choices=["rle", "polygon"],
        help="conversion mode: rle or polygon.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_COCO_CONFIG,
        help="Configuration for COCO categories",
    )
    parser.add_argument(
        "-mb",
        "--mask-base",
        help="Path to the BitMasks base folder.",
    )
    parser.add_argument(
        "-om",
        "--only-mask",
        action="store_true",
        help="Path to the BitMasks base folder.",
    )
    return parser.parse_args()


def bitmasks_loader(mask_name: str) -> List[InstanceType]:
    """Parse instances from the bitmask."""
    bitmask = np.asarray(Image.open(mask_name)).astype(np.int32)
    category_map = bitmask[:, :, 0]
    attributes_map = bitmask[:, :, 1]
    instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

    instances: List[InstanceType] = []

    # 0 is for the background
    instance_ids = np.sort(np.unique(instance_map[instance_map >= 1]))
    for instance_id in instance_ids:
        mask_inds_i = instance_map == instance_id
        attributes_i = np.unique(attributes_map[mask_inds_i])
        category_ids_i = np.unique(category_map[mask_inds_i])

        assert attributes_i.shape[0] == 1
        assert category_ids_i.shape[0] == 1
        attribute = attributes_i[0]
        category_id = category_ids_i[0]

        mask = mask_inds_i.astype(np.int32)
        bbox = mask_to_bbox(mask)
        area = np.sum(mask).tolist()

        instance = InstanceType(
            instance_id=int(instance_id),
            category_id=int(category_id),
            truncated=bool(attribute & (1 << 3)),
            occluded=bool(attribute & (1 << 2)),
            crowd=bool(attribute & (1 << 1)),
            ignore=bool(attribute & (1 << 1)),
            mask=mask,
            bbox=bbox,
            area=area,
        )
        instances.append(instance)

    return instances


def bitmask2coco_wo_ids(
    image_id: int, mask_name: str, mask_mode: str = "rle"
) -> List[AnnType]:
    """Convert bitmasks annotations of an image to RLEs or polygons."""
    instances: List[InstanceType] = bitmasks_loader(mask_name)
    annotations: List[AnnType] = []
    for instance in instances:
        annotation = AnnType(
            id=0,  # set further
            image_id=image_id,
            category_id=instance["category_id"],
            instance_id=instance["instance_id"],
            iscrowd=instance["crowd"],
            ignore=instance["ignore"],
        )
        annotation = set_seg_object_geometry(
            annotation, instance["mask"], mask_mode
        )
        annotations.append(annotation)
    return annotations


def bitmask2coco_wo_ids_parallel(
    image_ids: List[int],
    mask_names: List[str],
    mask_mode: str = "rle",
    nproc: int = 4,
) -> List[AnnType]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        annotations_list = pool.starmap(
            partial(bitmask2coco_wo_ids, mask_mode=mask_mode),
            tqdm(
                zip(image_ids, mask_names),
                total=len(image_ids),
            ),
        )
    annotations: List[AnnType] = []
    for anns in annotations_list:
        annotations.extend(anns)

    annotations = sorted(annotations, key=lambda ann: ann["image_id"])
    for i, annotation in enumerate(annotations):
        ann_id = i + 1
        annotation["id"] = ann_id
    return annotations


def bitmask2coco_with_ids(
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


def bitmask2coco_with_ids_parallel(
    annotations_list: List[List[AnnType]],
    mask_names: List[str],
    category_ids_list: List[List[int]],
    instance_ids_list: List[List[int]],
    mask_mode: str = "rle",
    nproc: int = 4,
) -> List[AnnType]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        annotations_list = pool.starmap(
            partial(bitmask2coco_with_ids, mask_mode=mask_mode),
            tqdm(
                zip(
                    annotations_list,
                    mask_names,
                    category_ids_list,
                    instance_ids_list,
                ),
                total=len(annotations_list),
            ),
        )
    annotations: List[AnnType] = []
    for anns in annotations_list:
        annotations.extend(anns)

    return annotations


def bitmask2coco_ins_seg(
    mask_base: str,
    shape: Tuple[int, int],
    files: List[str],
    categories: List[CatType],
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    images: List[ImgType] = []
    image_ids: List[int] = []
    mask_names: List[str] = []

    logger.info("Collecting bitmasks...")

    image_id = 0
    for file_ in tqdm(files):
        image_id += 1
        image = ImgType(
            id=image_id,
            file_name=file_.replace(".png", ".jpg"),
            height=shape[0],
            width=shape[1],
        )
        images.append(image)

        image_ids.append(image_id)
        mask_name = os.path.join(mask_base, file_)
        mask_names.append(mask_name)

    annotations = bitmask2coco_wo_ids_parallel(
        image_ids, mask_names, mask_mode, nproc
    )
    return GtType(
        type="instances",
        categories=categories,
        images=images,
        annotations=annotations,
    )


def bitmask2coco_seg_track(
    mask_base: str,
    shape: Tuple[int, int],
    all_files: List[str],
    categories: List[CatType],
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    files_list = group_and_sort_files(all_files)
    videos: List[VidType] = []
    images: List[ImgType] = []
    image_ids: List[int] = []
    mask_names: List[str] = []

    logger.info("Collecting bitmasks...")

    video_id, image_id = 0, 0
    for files in files_list:
        video_name = os.path.split(files[0])[0]
        video_id += 1
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for frame_id, file_ in tqdm(enumerate(files)):
            image_id += 1
            image = ImgType(
                video_id=video_id,
                frame_id=frame_id,
                id=image_id,
                file_name=file_.replace(".png", ".jpg"),
                height=shape[0],
                width=shape[1],
            )
            images.append(image)

            image_ids.append(image_id)
            mask_name = os.path.join(mask_base, file_.replace(".jpg", ".png"))
            mask_names.append(mask_name)

    annotations = bitmask2coco_wo_ids_parallel(
        image_ids, mask_names, mask_mode, nproc
    )
    return GtType(
        type="instances",
        categories=categories,
        videos=videos,
        images=images,
        annotations=annotations,
    )


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
    image_id, ann_id = 0, 0
    images: List[ImgType] = []

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[AnnType]] = []

    logger.info("Collecting annotations...")

    for image_anns in tqdm(frames):
        instance_id = 0
        image_id += 1
        image = ImgType(
            id=image_id,
            file_name=image_anns.name,
            height=shape[0],
            width=shape[1],
        )
        images.append(image)

        mask_name = os.path.join(
            mask_base,
            # Bitmask in .png format, image in .jpg format
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
            if category_ignored and remove_ignore:
                continue

            ann_id += 1
            instance_id += 1
            iscrowd, ignore = get_bdd100k_object_attributes(
                label, category_ignored
            )
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

        category_ids_list.append(category_ids)
        instance_ids_list.append(instance_ids)
        annotations_list.append(annotations)

    annotations = bitmask2coco_with_ids_parallel(
        annotations_list,
        mask_names,
        category_ids_list,
        instance_ids_list,
        mask_mode,
        nproc,
    )

    return GtType(
        type="instances",
        categories=categories,
        images=images,
        annotations=annotations,
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
    video_id, image_id, ann_id = 0, 0, 0

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[AnnType]] = []

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_name = video_anns[0].video_name
        video_id += 1
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in video_anns:
            image_id += 1
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
                # Bitmask in .png format, image in .jpg format
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
                instance_id, global_instance_id = get_bdd100k_instance_id(
                    instance_id_maps, global_instance_id, scalabel_id
                )

                ann_id += 1
                iscrowd, ignore = get_bdd100k_object_attributes(
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

            category_ids_list.append(category_ids)
            instance_ids_list.append(instance_ids)
            annotations_list.append(annotations)

    annotations = bitmask2coco_with_ids_parallel(
        annotations_list,
        mask_names,
        category_ids_list,
        instance_ids_list,
        mask_mode,
        nproc,
    )

    return GtType(
        type="instances",
        categories=categories,
        videos=videos,
        images=images,
        annotations=annotations,
    )


def start_converting(args: argparse.Namespace) -> List[Frame]:
    """Logs settings and load annoatations."""
    logger.info(
        "Mode: %s\nremove-ignore: %s\nignore-as-class: %s",
        args.mode,
        args.remove_ignore,
        args.ignore_as_class,
    )
    logger.info("Loading annotations...")
    labels = load(args.label, args.nproc)
    logger.info("Start format converting...")

    return labels


def main() -> None:
    """Main function."""
    args = parse_args()
    assert args.mode in ["det", "box_track", "ins_seg", "seg_track"]
    categories, name_mapping, ignore_mapping = load_coco_config(
        mode=args.mode,
        filepath=args.config,
        ignore_as_class=args.ignore_as_class,
    )

    shape = (args.height, args.width)
    if args.only_mask:
        assert args.mode in ["ins_seg", "seg_track"]
        convert_function = dict(
            ins_seg=bitmask2coco_ins_seg,
            seg_track=bitmask2coco_seg_track,
        )[args.mode]
        coco = convert_function(
            args.label,
            shape,
            list_files(args.label, suffix=".png"),
            categories,
            args.mask_mode,
            args.nproc,
        )
    else:
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
                mask_base=args.mask_base,
                mask_mode=args.mask_mode,
                nproc=args.nproc,
            )
        frames = start_converting(args)
        coco = convert_func(
            shape=shape,
            frames=frames,
            categories=categories,
            name_mapping=name_mapping,
            ignore_mapping=ignore_mapping,
            ignore_as_class=args.ignore_as_class,
            remove_ignore=args.remove_ignore,
        )

    logger.info("Saving converted annotations to disk...")
    with open(args.output, "w") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
