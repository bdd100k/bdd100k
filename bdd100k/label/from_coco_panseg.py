"""Convert COCO panoptic segmentation format from BDD100K bitmasks."""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayI32, NDArrayU8
from scalabel.label.coco_typing import PanopticAnnType, PanopticGtType
from tqdm import tqdm

from ..common.logger import logger


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="coco panoptic format to bdd100k bitmasks"
    )
    parser.add_argument("-i", "--input", help="path to the json file.")
    parser.add_argument(
        "-o",
        "--output",
        help="path to folder of panoptic segmentation bitmask",
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
        help="Path to the input panoptic segmentation mask folder.",
    )
    return parser.parse_args()


def panseg2bitmask(
    annotation: PanopticAnnType, pan_mask_base: str, mask_base: str
) -> None:
    """Convert COCO panoptic annotations of an image to BDD100K format."""
    pan_name = os.path.join(pan_mask_base, annotation["file_name"])
    pan_fmt: NDArrayI32 = np.asarray(Image.open(pan_name)).astype(
        dtype=np.int32
    )
    instance_map = pan_fmt[..., 0] + (pan_fmt[..., 1] << 8)

    height, width = pan_fmt.shape[:2]
    bitmask: NDArrayU8 = np.zeros((height, width, 4), dtype=np.uint8)
    bitmask[..., 3] = pan_fmt[..., 0]
    bitmask[..., 2] = pan_fmt[..., 1]

    for segm_info in annotation["segments_info"]:
        instance_id = segm_info["id"]
        cur_mask = instance_map == instance_id
        category_id = segm_info["category_id"]
        bitmask[..., 0] = (
            bitmask[..., 0] * (1 - cur_mask) + category_id * cur_mask
        )
        attributes = (segm_info["iscrowd"] << 1) + segm_info["ignore"]
        bitmask[..., 1] = (
            bitmask[..., 1] * (1 - cur_mask) + attributes * cur_mask
        )

    mask_name = os.path.join(mask_base, annotation["file_name"])
    pil_bitmask = Image.fromarray(bitmask)
    pil_bitmask.save(mask_name)


def coco_pan_seg2bitmask(
    coco_pan_seg: PanopticGtType,
    pan_mask_base: str,
    mask_base: str,
    nproc: int = NPROC,
) -> None:
    """Converting COCO panoptic segmentation dataset to BDD100K format."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        pool.map(
            partial(
                panseg2bitmask,
                pan_mask_base=pan_mask_base,
                mask_base=mask_base,
            ),
            tqdm(coco_pan_seg["annotations"]),
        )


def main() -> None:
    """Main function."""
    args = parse_args()
    with open(args.input, encoding="utf-8") as fp:
        coco_pan_seg = json.load(fp)

    logger.info("Start format converting...")
    coco_pan_seg2bitmask(
        coco_pan_seg, args.pan_mask_base, args.output, args.nproc
    )
    logger.info("Finished!")


if __name__ == "__main__":
    main()
