import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scalabel.label.coco_typing import PnpGtType, PnpAnnType
from tqdm import tqdm

from ..common.logger import logger


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="bdd100k to coco format")
    parser.add_argument("-i", "--input", help="path to the json file.")
    parser.add_argument(
        "-o",
        "--output",
        help="path to folder of panoptic segmentation bitmask",
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


def panoptic2bitmask(
    annotation: PnpAnnType, panoptic_base: str, mask_base: str
) -> None:
    """Convert coco panoptic annotations of an image to BDD100K format."""
    panoptic_name = os.path.join(panoptic_base, annotation["file_name"])
    pan_fmt = np.array(Image.open(panoptic_name), dtype=np.uint32)
    instance_map = pan_fmt[..., 0] + (pan_fmt[..., 1] << 8)

    height, width = pan_fmt.shape[:2]
    bitmask = np.zeros((height, width, 4), dtype=np.uint8)
    bitmask[..., 3] = pan_fmt[..., 0]
    bitmask[..., 2] = pan_fmt[..., 1]

    for segm_info in annotation["segments_info"]:
        instance_id = segm_info["id"]
        category_id = segm_info["category_id"]
        cur_mask = instance_map == instance_id
        bitmask[..., 0] = (
            bitmask[..., 0] * (1 - cur_mask) + category_id * cur_mask
        )
        attributes = (segm_info["iscrowd"] << 1) + segm_info["ignore"]
        bitmask[..., 1] = (
            bitmask[..., 1] * (1 - cur_mask) + attributes * cur_mask
        )

    mask_name = os.path.join(mask_base, annotation["file_name"])
    bitmask = Image.fromarray(bitmask)
    bitmask.save(mask_name)


def coco_panoptic_seg2bitmask(
    coco_panoptic: PnpGtType,
    panoptic_base: str,
    mask_base: str,
    nproc: int = 4,
) -> None:
    """Converting COCO panoptic segmentation to BDD100K format."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        pool.map(
            partial(
                panoptic2bitmask,
                panoptic_base=panoptic_base,
                mask_base=mask_base,
            ),
            tqdm(coco_panoptic["annotations"]),
        )


def main() -> None:
    """Main function."""
    args = parse_args()
    with open(args.input) as fp:
        coco_panoptic = json.load(fp)

    logger.info("Start format converting...")
    coco_panoptic_seg2bitmask(
        coco_panoptic, args.panoptic_base, args.output, args.nproc
    )
    logger.info("Finished!")


if __name__ == "__main__":
    main()
