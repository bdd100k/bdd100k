"""Convert BDD100K bitmasks to COCO panoptic segmentation format."""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
from PIL import Image
from scalabel.label.coco_typing import (
    ImgType,
    PnpCatType,
    PnpGtType,
    PnpAnnType,
    PnpSegType,
)
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import list_files
from .label import labels
from .to_coco import bitmasks_loader


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="bdd100k to coco format")
    parser.add_argument("-i", "--input", help="path to the bitmask folder.")
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco panoptic formatted label file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "-pb",
        "--panoptic-base",
        help="Path to the panoptic segmentation mask folder.",
    )
    return parser.parse_args()


def bitmask2pan_mask(mask_name: str, panoptic_name) -> None:
    """Convert bitmask into panoptic segmentation mask."""
    bitmask = np.asarray(Image.open(mask_name)).astype(np.int32)
    height, width = bitmask.shape[:2]

    pan_fmt = np.zeros((height, width, 3), dtype=np.uint8)
    pan_fmt[..., 0] = bitmask[..., 3]
    pan_fmt[..., 1] = bitmask[..., 2]

    pan_mask = Image.fromarray(pan_fmt)
    pan_mask.save(panoptic_name)


def bitmask2pan_json(image, image_id, mask_name) -> PnpAnnType:
    """Convert bitmask into panoptic segmentation json."""
    instances, img_shape = bitmasks_loader(mask_name)
    image["height"] = img_shape.height
    image["width"] = img_shape.width

    cat_id_to_idx: Dict[int, int] = dict()

    segments_info: List[PnpSegType] = []
    for instance in instances:
        category_id = instance["category_id"]
        if category_id not in cat_id_to_idx:
            segment_info = PnpSegType(
                id=instance["instance_id"],  # set further
                category_id=category_id,
                area=instance["area"],
                iscrowd=instance["crowd"] or instance["ignored"],
                ignore=0,
            )
            segments_info.append(segment_info)
            cat_id_to_idx[category_id] = len(segment_info) - 1
        else:
            segment_info = segments_info[cat_id_to_idx[category_id]]
    annotation = PnpAnnType(
        image_id=image_id,
        file_name=image["file_name"],
        segments_info=segments_info,
    )
    return annotation


def bitmask2panoptic_parallel(
    mask_base: str,
    panoptic_base: str,
    images: List[ImgType],
    image_ids: List[int],
    nproc: int = 4,
) -> List[PnpAnnType]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    mask_names = [
        os.path.join(mask_base, image["file_name"]) for image in images
    ]
    panoptic_names = [
        os.path.join(panoptic_base, image["file_name"]) for image in images
    ]

    with Pool(nproc) as pool:
        pool.starmap(
            bitmask2pan_mask,
            tqdm(zip(mask_names, panoptic_names), total=len(mask_names)),
        )
        annotations = pool.starmap(
            bitmask2pan_json,
            tqdm(zip(images, image_ids, mask_names), total=len(images)),
        )
    annotations = sorted(annotations, key=lambda ann: ann["image_id"])
    return annotations


def bitmask2coco_panoptic_seg(
    mask_base: str,
    panoptic_base: str,
    nproc: int = 4,
) -> PnpGtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    files = list_files(mask_base, suffix=".png")

    images: List[ImgType] = []
    image_ids: List[int] = []

    logger.info("Collecting bitmasks...")

    image_id = 0
    for file_ in tqdm(files):
        image_id += 1
        image = ImgType(id=image_id, file_name=file_)
        images.append(image)
        image_ids.append(image_id)

    annotations = bitmask2panoptic_parallel(
        mask_base, panoptic_base, images, image_ids, nproc
    )
    categories: List[PnpCatType] = [
        PnpCatType(
            id=label.id,
            name=label.name,
            supercategory=label.category,
            isthing=label.hasInstances,
            color=label.color,
        )
        for label in labels
    ]
    return PnpGtType(
        categories=categories,
        images=images,
        annotations=annotations,
    )


def main() -> None:
    """Main function."""
    args = parse_args()

    logger.info("Start format converting...")
    coco = bitmask2coco_panoptic_seg(
        args.input, args.panoptic_base, args.nproc
    )
    logger.info("Saving converted annotations to disk...")
    with open(args.output, "w") as fp:
        json.dump(coco, fp)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
