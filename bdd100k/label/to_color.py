"""Convert poly2d to mask/bitmask.

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
import os
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import group_and_sort_files, list_files
from .palette import get_palette


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="masks/bitmasks to colormaps")
    parser.add_argument(
        "-l", "--label", help="path to the directory of masks/bitmasks."
    )
    parser.add_argument(
        "-o", "--output", help="path to save generated colormaps."
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["sem_seg", "drivable", "lane_mark", "ins_seg", "seg_track"],
        help="conversion mode.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for conversion.",
    )
    return parser.parse_args()


def mask_to_color(bitmask_file: str, colormap_file: str, mode: str) -> None:
    """Convert mask/bitmask to colormap for one image."""
    bitmask = Image.open(bitmask_file)
    if mode in ["ins_seg", "seg_track"]:
        bitmask = bitmask.split()[3]
    elif mode == "lane_mark":
        array = np.asarray(bitmask)
        # 15 = (1 << 4) - 1, only take the last 4 bits
        array = array & 15
        bitmask = Image.fromarray(array)
    palette = get_palette(mode)
    bitmask.putpalette(palette)
    bitmask.save(colormap_file)


def masks_to_colors(
    bitmasks_files: List[str],
    colormap_files: List[str],
    mode: str,
    nproc: int,
) -> None:
    """Convert mask/bitmask to colormap for a list of images."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        pool.starmap(
            partial(mask_to_color, mode=mode),
            tqdm(
                zip(bitmasks_files, colormap_files),
                total=len(bitmasks_files),
            ),
        )


def image_dataset_to_colormap(
    in_base: str,
    out_base: str,
    mode: str,
    nproc: int,
) -> None:
    """Convert instance segmentation bitmasks to labelmap."""
    if not os.path.isdir(out_base):
        os.makedirs(out_base)
    files = list_files(in_base, ".png")
    bitmasks_files: List[str] = []
    colormap_files: List[str] = []

    logger.info("Preparing annotations for image dataset to Colormap")

    for file_name in tqdm(files):
        label_path = os.path.join(in_base, file_name)
        color_path = os.path.join(out_base, file_name)
        bitmasks_files.append(label_path)
        colormap_files.append(color_path)
    masks_to_colors(bitmasks_files, colormap_files, mode, nproc)


def video_dataset_to_colormap(
    in_base: str,
    out_base: str,
    mode: str,
    nproc: int,
) -> None:
    """Convert segmentation tracking bitmasks to labelmap."""
    if not os.path.isdir(out_base):
        os.makedirs(out_base)
    files = list_files(in_base, ".png")
    files_list = group_and_sort_files(files)

    bitmasks_files: List[str] = []
    colormap_files: List[str] = []

    logger.info("Preparing annotations for video dataset to Colormap")

    for files in tqdm(files_list):
        assert len(files) > 0
        video_name = os.path.split(files[0])[0]
        video_path = os.path.join(out_base, video_name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)

        for file_name in files:
            label_path = os.path.join(in_base, file_name)
            color_path = os.path.join(out_base, file_name)
            bitmasks_files.append(label_path)
            colormap_files.append(color_path)
    masks_to_colors(bitmasks_files, colormap_files, mode, nproc)


def main() -> None:
    """Main function."""
    args = parse_args()
    colormap_func = (
        video_dataset_to_colormap
        if args.mode == "seg_track"
        else image_dataset_to_colormap
    )
    colormap_func(args.label, args.output, args.mode, args.nproc)


if __name__ == "__main__":
    main()
