"""Convert poly2d to bitmasks.

The annotation files in BDD100K format has additional annotaitons
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
from typing import List, Tuple

import matplotlib.patches as mpatches  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.path import Path  # type: ignore
from PIL import Image
from tqdm import tqdm

from ..common.iterator import VideoLabelIterator
from ..common.logger import logger
from ..common.typing import DictAny, ListAny
from ..common.utils import list_files
from .to_coco import parser_definition, start_converting


def parser_definition_bitmasks() -> argparse.ArgumentParser:
    """Definition of the parser."""
    parser = parser_definition()
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
        edgecolor=color,  # if not closed else 'none',
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


class SegTrack2BitMaskIterator(VideoLabelIterator):
    """Iterator for conversion of segtrack to bitmasks."""

    def __init__(
        self,
        out_base: str,
        nproc: int,
        ignore_as_class: bool = False,
        remove_ignore: bool = False,
    ):
        """Initialize the segtrack2bitmask iterator."""
        super().__init__("track", ignore_as_class, remove_ignore)
        self.out_base = out_base
        self.nproc = nproc

        self.out_paths: List[str] = []
        self.colors_list: List[ListAny] = []
        self.poly2ds_list: List[ListAny] = []

        self.colors: List[np.ndarray] = []
        self.poly2ds: ListAny = []

    def video_iteration(self, labels_per_video: List[DictAny]) -> None:
        """Actions for the video iteration."""
        assert len(labels_per_video) > 0
        video_name = labels_per_video[0]["video_name"]
        out_dir = os.path.join(self.out_base, video_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    def image_iteration(self, labels_per_image: DictAny) -> None:
        """Actions for the image iteration."""
        super().image_iteration(labels_per_image)
        video_name = labels_per_image["video_name"]
        out_name = labels_per_image["name"].replace(".jpg", ".png")
        out_path = os.path.join(self.out_base, video_name, out_name)
        self.out_paths.append(out_path)

        self.colors = []
        self.poly2ds = []

    def object_iteration(self, labels_per_object: DictAny) -> None:
        """Actions for the object iteration."""
        super().object_iteration(labels_per_object)
        truncated = int(bool(labels_per_object["attributes"]["Truncated"]))
        occluded = int(bool(labels_per_object["attributes"]["Occluded"]))
        crowd = int(bool(labels_per_object["attributes"]["Crowd"]))
        ignore = int(bool(labels_per_object["category_ignored"]))
        color = np.array(
            [
                labels_per_object["category_id"] & 255,
                (truncated << 3) + (occluded << 2) + (crowd << 1) + ignore,
                labels_per_object["instance_id"] >> 8,
                labels_per_object["instance_id"] & 255,
            ],
            dtype=np.uint8,
        )
        self.colors.append(color)
        self.poly2ds.append(labels_per_object["poly2d"])

    def after_iteration(self) -> DictAny:
        """Actions after the iteration."""
        logger.info("Converting annotations...")

        pool = Pool(self.nproc)
        pool.starmap(
            poly2d2bitmasks_per_image,
            tqdm(
                zip(self.out_paths, self.colors_list, self.poly2ds_list),
                total=len(self.out_paths),
            ),
        )
        pool.close()
        return self.coco

    def __call__(self, labels: List[List[DictAny]]) -> DictAny:
        """Executes iterations."""
        for labels_per_video in tqdm(labels):
            self.video_iteration(labels_per_video)
            for labels_per_image in labels_per_video:
                self.image_iteration(labels_per_image)
                for labels_per_object in labels_per_image["labels"]:
                    self.object_iteration(labels_per_object)
                self.colors_list.append(self.colors)
                self.poly2ds_list.append(self.poly2ds)
        return self.after_iteration()


def bitmask2labelmap_per_img(bitmask_file: str, colormap_file: str) -> None:
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


def bitmask2labelmap(out_base: str, color_base: str, nproc: int) -> None:
    """Convert BitMasks to labelmap."""
    if not os.path.isdir(color_base):
        os.makedirs(color_base)
    files_list = list_files(out_base)
    bitmasks_files = []
    colormap_files = []
    for files in files_list:
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

    pool = Pool(nproc)
    pool.starmap(
        bitmask2labelmap_per_img,
        tqdm(
            zip(bitmasks_files, colormap_files),
            total=len(bitmasks_files),
        ),
    )


def main() -> None:
    """Main function."""
    args, labels = start_converting()
    if args.mode == "seg_track":
        iterator = SegTrack2BitMaskIterator(
            args.out_path,
            args.nproc,
            args.ignore_as_class,
            args.remove_ignore,
        )
        iterator(labels)
    if args.colormap:
        bitmask2labelmap(args.out_path, args.color_path, args.nproc)

    logger.info("Finished!")


if __name__ == "__main__":
    main()
