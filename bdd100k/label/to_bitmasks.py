"""Convert poly2d to bitmasks.

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
from multiprocessing import Pool
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.path import Path  # type: ignore
from PIL import Image
from tqdm import tqdm

from ..common.logger import logger
from ..common.typing import DictAny, ListAny
from ..common.utils import init, list_files
from .to_coco import (
    get_instance_id,
    parser_definition_coco,
    process_category,
    start_converting,
)


def parser_definition_bitmasks() -> argparse.ArgumentParser:
    """Definition of the parser."""
    parser = parser_definition_coco()
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for mot evaluation",
    )
    parser.add_argument(
        "-cm",
        "--colormap",
        action="store_true",
        help="Save the colorized labels for MOTS.",
    )
    parser.add_argument(
        "-cp",
        "--color-path",
        default="/output/path",
        help="Path to save colorized images.",
    )
    return parser


def poly2patch(
    vertices: List[List[float]],
    types: str,
    color: Tuple[float, float, float],
    closed: bool,
) -> mpatches.PathPatch:
    """Draw polygons using the Bezier curve."""
    moves = {"L": Path.LINETO, "C": Path.CURVE4}
    points = list(vertices)
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.LINETO)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else "none",
        edgecolor=color,
        lw=0 if closed else 1,
        alpha=1,
        antialiased=False,
        snap=True,
    )


def poly2d2bitmasks_per_image(
    out_path: str,
    colors: List[np.ndarray],
    poly2ds: ListAny,
) -> None:
    """Converting seg_track poly2d to bitmasks for a video."""
    assert len(colors) == len(poly2ds)
    shape = np.array([720, 1280])

    fig = plt.figure(facecolor="0")
    fig.set_size_inches(shape[::-1] / fig.get_dpi())
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for i, poly2d in enumerate(poly2ds):
        for poly in poly2d:
            ax.add_patch(
                poly2patch(
                    poly["vertices"],
                    poly["types"],
                    # 0 / 255.0 for the background
                    color=((i + 1) / 255.0, (i + 1) / 255.0, (i + 1) / 255.0),
                    closed=True,
                )
            )

    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    out = out.reshape((*shape, -1))[..., 0]
    plt.close()

    img = np.zeros((*shape, 4), dtype=np.uint8)
    for i, color in enumerate(colors):
        # 0 is for the background
        img[out == i + 1] = color
    img = Image.fromarray(img)
    img.save(out_path)


def set_color(
    label: DictAny, category_id: int, ann_id: int, category_ignored: bool
) -> np.ndarray:
    """Set the color for an instance given its attributes and ID."""
    attributes = label["attributes"]
    truncated = int(attributes.get("Truncated", False))
    occluded = int(attributes.get("Occluded", False))
    crowd = int(attributes.get("Crowd", False))
    ignore = int(category_ignored)
    color = np.array(
        [
            category_id & 255,
            (truncated << 3) + (occluded << 2) + (crowd << 1) + ignore,
            ann_id >> 8,
            ann_id & 255,
        ],
        dtype=np.uint8,
    )
    return color


def bitmask_conversion(
    nproc: int,
    out_paths: List[str],
    colors_list: List[ListAny],
    poly2ds_list: List[ListAny],
) -> None:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    pool = Pool(nproc)
    pool.starmap(
        poly2d2bitmasks_per_image,
        tqdm(
            zip(out_paths, colors_list, poly2ds_list),
            total=len(out_paths),
        ),
    )
    pool.close()


def insseg2bitmasks(
    labels: List[List[DictAny]],
    out_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    nproc: int = 4,
) -> None:
    """Converting seg_track poly2d to bitmasks."""
    assert len(labels) == 1
    os.makedirs(out_base, exist_ok=True)

    _, cat_name2id = init(mode="track", ignore_as_class=ignore_as_class)

    out_paths: List[str] = []
    colors_list: List[ListAny] = []
    poly2ds_list: List[ListAny] = []

    logger.info("Preparing annotations for InsSeg to Bitmasks")

    for image_anns in tqdm(labels[0]):
        ann_id = 1

        image_name = image_anns["name"].replace(".jpg", ".png")
        image_name = os.path.split(image_name)[-1]
        out_path = os.path.join(out_base, image_name)
        out_paths.append(out_path)

        colors: List[np.ndarray] = []
        poly2ds: ListAny = []

        for label in image_anns["labels"]:
            if "poly2d" not in label:
                continue

            category_ignored, category_id = process_category(
                label["category"], ignore_as_class, cat_name2id
            )
            if remove_ignore and category_ignored:
                continue
            label["category_ignored"] = category_ignored

            color = set_color(label, category_id, ann_id, category_ignored)
            colors.append(color)
            poly2ds.append(label["poly2d"])
            ann_id += 1

        colors_list.append(colors)
        poly2ds_list.append(poly2ds)

    bitmask_conversion(nproc, out_paths, colors_list, poly2ds_list)


def segtrack2bitmasks(
    labels: List[List[DictAny]],
    out_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    nproc: int = 4,
) -> None:
    """Converting seg_track poly2d to bitmasks."""
    _, cat_name2id = init(mode="track", ignore_as_class=ignore_as_class)

    out_paths: List[str] = []
    colors_list: List[ListAny] = []
    poly2ds_list: List[ListAny] = []

    logger.info("Preparing annotations for SegTrack to Bitmasks")

    for video_anns in tqdm(labels):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_name = video_anns[0]["video_name"]
        out_dir = os.path.join(out_base, video_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        for image_anns in video_anns:
            image_name = image_anns["name"].replace(".jpg", ".png")
            image_name = os.path.split(image_name)[-1]
            out_path = os.path.join(out_dir, image_name)
            out_paths.append(out_path)

            colors: List[np.ndarray] = []
            poly2ds: ListAny = []

            for label in image_anns["labels"]:
                if "poly2d" not in label:
                    continue

                category_ignored, category_id = process_category(
                    label["category"], ignore_as_class, cat_name2id
                )
                if category_ignored and remove_ignore:
                    continue
                label["category_ignored"] = category_ignored

                bdd100k_id = str(label["id"])
                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, bdd100k_id
                )

                color = set_color(
                    label, category_id, instance_id, category_ignored
                )
                colors.append(color)
                poly2ds.append(label["poly2d"])

            colors_list.append(colors)
            poly2ds_list.append(poly2ds)

    bitmask_conversion(nproc, out_paths, colors_list, poly2ds_list)


def bitmask2labelmap_per_image(bitmask_file: str, colormap_file: str) -> None:
    """Convert BitMasks to labelmap for one image."""
    bitmask = np.asarray(Image.open(bitmask_file))
    colormap = np.zeros((*bitmask.shape[:2], 3), dtype=bitmask.dtype)
    # For category. 8 * 30 = 240 < 255.
    colormap[..., 0] = bitmask[..., 0] * 30
    # For instance id.
    colormap[..., 1] = bitmask[..., 2]
    colormap[..., 2] = bitmask[..., 3]

    img = Image.fromarray(colormap)
    img.save(colormap_file)


def insseg2colormap(out_base: str, color_base: str, nproc: int) -> None:
    """Convert instance segmentation bitmasks to labelmap."""
    if not os.path.isdir(color_base):
        os.makedirs(color_base)
    files_list = os.listdir(out_base)
    bitmasks_files: List[str] = []
    colormap_files: List[str] = []

    logger.info("Preparing annotations for InsSeg to Colormap")

    for file_name in tqdm(files_list):
        if not file_name.endswith(".png"):
            continue
        label_path = os.path.join(out_base, file_name)
        color_path = os.path.join(color_base, file_name)
        bitmasks_files.append(label_path)
        colormap_files.append(color_path)
    colormap_conversion(nproc, bitmasks_files, colormap_files)


def segtrack2colormap(out_base: str, color_base: str, nproc: int) -> None:
    """Convert segmentation tracking bitmasks to labelmap."""
    if not os.path.isdir(color_base):
        os.makedirs(color_base)
    files_list = list_files(out_base)
    bitmasks_files: List[str] = []
    colormap_files: List[str] = []

    logger.info("Preparing annotations for SegTrack to Colormap")

    for files in tqdm(files_list):
        assert len(files) > 0
        video_name = files[0].rsplit("/", 3)[-2]
        video_path = os.path.join(color_base, video_name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        for file_name in files:
            image_name = os.path.split(file_name)[-1]
            save_path = os.path.join(color_base, video_name, image_name)
            bitmasks_files.append(file_name)
            colormap_files.append(save_path)
    colormap_conversion(nproc, bitmasks_files, colormap_files)


def colormap_conversion(
    nproc: int,
    bitmasks_files: List[str],
    colormap_files: List[str],
) -> None:
    """Execute the colormap conversion in parallel."""
    logger.info("Converting annotations...")

    pool = Pool(nproc)
    pool.starmap(
        bitmask2labelmap_per_image,
        tqdm(
            zip(bitmasks_files, colormap_files),
            total=len(bitmasks_files),
        ),
    )


def main() -> None:
    """Main function."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # matplotlib offscreen render
    args, labels = start_converting(parser_definition_bitmasks)
    bitmask_func = dict(ins_seg=insseg2bitmasks, seg_track=segtrack2bitmasks)[
        args.mode
    ]
    bitmask_func(
        labels,
        args.out_path,
        args.ignore_as_class,
        args.remove_ignore,
        args.nproc,
    )

    colormap_func = dict(ins_seg=insseg2colormap, seg_track=segtrack2colormap)[
        args.mode
    ]
    if args.colormap:
        colormap_func(args.out_path, args.color_path, args.nproc)

    logger.info("Finished!")


if __name__ == "__main__":
    main()
