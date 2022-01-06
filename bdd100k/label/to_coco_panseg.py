"""Convert BDD100K bitmasks to COCO panoptic segmentation format."""

import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayI32, NDArrayU8
from scalabel.label.coco_typing import (
    ImgType,
    PanopticAnnType,
    PanopticCatType,
    PanopticGtType,
    PanopticSegType,
)
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import list_files
from .label import labels
from .to_coco import bitmasks_loader
from .to_mask import STUFF_NUM


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="bdd100k bitmasks to coco panoptic format"
    )
    parser.add_argument("-i", "--input", help="path to the bitmask folder.")
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco panoptic formatted json file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "-pb",
        "--pan-mask-base",
        help="Path to the output panoptic segmentation mask folder.",
    )
    return parser.parse_args()


def bitmask2pan_mask(mask_name: str, pan_name: str) -> None:
    """Convert bitmask into panoptic segmentation mask."""
    bitmask: NDArrayI32 = np.asarray(Image.open(mask_name)).astype(
        dtype=np.int32
    )
    height, width = bitmask.shape[:2]

    pan_fmt: NDArrayU8 = np.zeros((height, width, 3), dtype=np.uint8)
    pan_fmt[..., 0] = bitmask[..., 3]
    pan_fmt[..., 1] = bitmask[..., 2]

    pan_mask = Image.fromarray(pan_fmt)
    pan_mask.save(pan_name)


def bitmask2pan_json(image: ImgType, mask_name: str) -> PanopticAnnType:
    """Convert bitmask into panoptic segmentation json."""
    instances, img_shape = bitmasks_loader(mask_name)
    image["height"] = img_shape.height
    image["width"] = img_shape.width

    cat_id_to_idx: Dict[int, int] = {}

    segments_info: List[PanopticSegType] = []
    for instance in instances:
        category_id = instance["category_id"]
        if category_id not in cat_id_to_idx:
            segment_info = PanopticSegType(
                id=instance["instance_id"],
                category_id=category_id,
                area=instance["area"],
                iscrowd=instance["crowd"] or instance["ignored"],
                ignore=0,
            )
            segments_info.append(segment_info)
            if category_id <= STUFF_NUM:
                cat_id_to_idx[category_id] = len(segment_info) - 1
        else:
            segment_info = segments_info[cat_id_to_idx[category_id]]
            segment_info["area"] += instance["area"]
            segment_info["iscrowd"] = 0
    annotation = PanopticAnnType(
        image_id=image["id"],
        file_name=image["file_name"].replace(".jpg", ".png"),
        segments_info=segments_info,
    )
    return annotation


def bitmask2panseg_parallel(
    mask_base: str,
    pan_mask_base: str,
    images: List[ImgType],
    nproc: int = NPROC,
) -> List[PanopticAnnType]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    mask_names = [
        os.path.join(mask_base, image["file_name"]) for image in images
    ]
    pan_names = [
        os.path.join(pan_mask_base, image["file_name"]) for image in images
    ]

    with Pool(nproc) as pool:
        pool.starmap(
            bitmask2pan_mask,
            tqdm(zip(mask_names, pan_names), total=len(mask_names)),
        )
        annotations = pool.starmap(
            bitmask2pan_json, tqdm(zip(images, mask_names), total=len(images))
        )
    annotations = sorted(annotations, key=lambda ann: ann["image_id"])
    return annotations


def bitmask2coco_pan_seg(
    mask_base: str,
    pan_mask_base: str,
    nproc: int = NPROC,
) -> PanopticGtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    files = list_files(mask_base, suffix=".png")
    images: List[ImgType] = []

    logger.info("Collecting bitmasks...")

    image_id = 0
    for file_ in tqdm(files):
        image_id += 1
        image = ImgType(id=image_id, file_name=file_)
        images.append(image)

    annotations = bitmask2panseg_parallel(
        mask_base, pan_mask_base, images, nproc
    )
    categories: List[PanopticCatType] = [
        PanopticCatType(
            id=label.id,
            name=label.name,
            supercategory=label.category,
            isthing=label.hasInstances,
            color=label.color,
        )
        for label in labels
    ]
    return PanopticGtType(
        categories=categories,
        images=images,
        annotations=annotations,
    )


def main() -> None:
    """Main function."""
    args = parse_args()

    logger.info("Start format converting...")
    coco = bitmask2coco_pan_seg(args.input, args.pan_mask_base, args.nproc)
    logger.info("Saving converted annotations to disk...")
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(coco, fp)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
