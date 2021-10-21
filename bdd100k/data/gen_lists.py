"""Generate the image lists for data loaders."""

import os
import sys
from os import path as osp

from ..common.logger import logger


def gen_list(
    data_root: str,
    data_dir: str,
    list_dir: str,
    phase: str,
    list_type: str,
    suffix: str = ".jpg",
) -> None:
    """Generate the list."""
    phase_dir = osp.join(data_root, data_dir, phase)
    if not osp.exists(phase_dir):
        raise ValueError(f"Can not find folder {phase_dir}")
    images = sorted(
        [
            osp.join(data_dir, phase, n)
            for n in os.listdir(phase_dir)
            if n[-len(suffix) :] == suffix
        ]
    )
    logger.info("Found %d items in %s %s", len(images), data_dir, phase)
    out_path = osp.join(list_dir, f"{phase}_{list_type}.txt")
    if not osp.exists(list_dir):
        os.makedirs(list_dir)
    logger.info("Writing %s", out_path)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(images))


def gen_images(
    data_root: str, list_dir: str, image_type: str = "100k"
) -> None:
    """Generate lists for different phases."""
    for phase in ["train", "val", "test"]:
        gen_list(
            data_root,
            osp.join("images", image_type),
            list_dir,
            phase,
            "images",
            ".jpg",
        )


def gen_drivable(data_root: str) -> None:
    """Generate lists for drivable area."""
    image_type = "100k"
    label_dir = "drivable_maps/labels"
    list_dir = "lists/100k/drivable"

    gen_images(data_root, list_dir, image_type)

    for p in ["train", "val"]:
        gen_list(
            data_root, label_dir, list_dir, p, "labels", "drivable_id.png"
        )


def gen_seg(data_root: str) -> None:
    """Generate lists for segmentation."""
    image_type = "10k"
    label_dir = "seg_maps/labels"
    list_dir = "lists/10k/seg"

    gen_images(data_root, list_dir, image_type)

    for p in ["train", "val"]:
        gen_list(data_root, label_dir, list_dir, p, "labels", "train_id.png")


if __name__ == "__main__":
    gen_drivable(sys.argv[1])
    gen_seg(sys.argv[1])
