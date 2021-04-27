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

import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from PIL import Image
from scalabel.label.io import group_and_sort
from scalabel.label.to_coco import load_coco_config, process_category
from scalabel.label.transforms import poly_to_patch
from scalabel.label.typing import Frame, Label, Poly2D
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import DEFAULT_COCO_CONFIG, get_bdd100k_instance_id
from .label import labels as SEMSEG_LABELS
from .to_coco import parse_args, start_converting

IGNORE_LABEL = 255
SHAPE = np.array([720, 1280])


def frame_to_mask(
    out_path: str,
    colors: List[np.ndarray],
    poly2ds: List[List[Poly2D]],
    with_instances: bool = True,
) -> None:
    """Converting a frame of poly2ds to mask/bitmask."""
    assert len(colors) == len(poly2ds)

    matplotlib.use("Agg")
    fig = plt.figure(facecolor="0")
    fig.set_size_inches(SHAPE[::-1] / fig.get_dpi())
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, SHAPE[1])
    ax.set_ylim(0, SHAPE[0])
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for i, poly2d in enumerate(poly2ds):
        for poly in poly2d:
            ax.add_patch(
                poly_to_patch(
                    poly.vertices,
                    poly.types,
                    # 0 / 255.0 for the background
                    color=(
                        ((i + 1) >> 8) / 255.0,
                        ((i + 1) % 255) / 255.0,
                        0.0,
                    ),
                    closed=True,
                )
            )

    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    out = out.reshape((*SHAPE, -1)).astype(np.int32)
    out = (out[..., 0] << 8) + out[..., 1]
    plt.close()

    if with_instances:
        img = np.zeros([*SHAPE, 4], dtype=np.uint8)
    else:
        img = np.ones([*SHAPE, 1], dtype=np.uint8) * IGNORE_LABEL

    for i, color in enumerate(colors):
        # 0 is for the background
        img[out == i + 1] = color
    pil_img = Image.fromarray(img.squeeze())
    pil_img.save(out_path)


def set_instance_color(
    label: Label, category_id: int, ann_id: int, category_ignored: bool
) -> np.ndarray:
    """Set the color for an instance given its attributes and ID."""
    attributes = label.attributes
    if attributes is None:
        truncated, occluded, crowd, ignore = 0, 0, 0, 0
    else:
        truncated = int(attributes.get("truncated", False))
        occluded = int(attributes.get("occluded", False))
        crowd = int(attributes.get("crowd", False))
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


def frames_to_masks(
    nproc: int,
    out_paths: List[str],
    colors_list: List[List[np.ndarray]],
    poly2ds_list: List[List[List[Poly2D]]],
    with_instances: bool = True,
) -> None:
    """Execute the mask conversion in parallel."""
    with Pool(nproc) as pool:
        pool.starmap(
            partial(frame_to_mask, with_instances=with_instances),
            tqdm(
                zip(out_paths, colors_list, poly2ds_list),
                total=len(out_paths),
            ),
        )


def semseg_to_masks(
    frames: List[Frame],
    out_base: str,
    ignore_as_class: bool = False,  # pylint: disable=unused-argument
    remove_ignore: bool = False,  # pylint: disable=unused-argument
    nproc: int = 4,
) -> None:
    """Converting semantic segmentation poly2d to 1-channel masks."""
    os.makedirs(out_base, exist_ok=True)

    out_paths: List[str] = []
    colors_list: List[List[np.ndarray]] = []
    poly2ds_list: List[List[List[Poly2D]]] = []

    cat_name2id = {label.name: label.trainId for label in SEMSEG_LABELS}

    logger.info("Preparing annotations for Semseg to Bitmasks")

    for image_anns in tqdm(frames):
        image_name = image_anns.name.replace(".jpg", ".png")
        image_name = os.path.split(image_name)[-1]
        out_path = os.path.join(out_base, image_name)
        out_paths.append(out_path)

        colors: List[np.ndarray] = []
        poly2ds: List[List[Poly2D]] = []

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.category not in cat_name2id:
                continue
            if label.poly_2d is None:
                continue

            category_id = cat_name2id[label.category]
            color = np.array([category_id])
            colors.append(color)
            poly2ds.append(label.poly_2d)

        colors_list.append(colors)
        poly2ds_list.append(poly2ds)

    frames_to_masks(
        nproc, out_paths, colors_list, poly2ds_list, with_instances=False
    )


def insseg_to_bitmasks(
    frames: List[Frame],
    out_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    nproc: int = 4,
) -> None:
    """Converting instance segmentation poly2d to bitmasks."""
    os.makedirs(out_base, exist_ok=True)

    categories, name_mapping, ignore_mapping = load_coco_config(
        mode="track",
        filepath=DEFAULT_COCO_CONFIG,
        ignore_as_class=ignore_as_class,
    )

    out_paths: List[str] = []
    colors_list: List[List[np.ndarray]] = []
    poly2ds_list: List[List[List[Poly2D]]] = []

    logger.info("Preparing annotations for InsSeg to Bitmasks")

    for image_anns in tqdm(frames):
        ann_id = 0

        image_name = image_anns.name.replace(".jpg", ".png")
        image_name = os.path.split(image_name)[-1]
        out_path = os.path.join(out_base, image_name)
        out_paths.append(out_path)

        colors: List[np.ndarray] = []
        poly2ds: List[List[Poly2D]] = []

        if image_anns.labels is None:
            continue

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

            ann_id += 1
            color = set_instance_color(
                label, category_id, ann_id, category_ignored
            )
            colors.append(color)
            poly2ds.append(label.poly_2d)

        colors_list.append(colors)
        poly2ds_list.append(poly2ds)

    frames_to_masks(nproc, out_paths, colors_list, poly2ds_list)


def segtrack_to_bitmasks(
    frames: List[Frame],
    out_base: str,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    nproc: int = 4,
) -> None:
    """Converting segmentation tracking poly2d to bitmasks."""
    frames_list = group_and_sort(frames)
    categories, name_mapping, ignore_mapping = load_coco_config(
        mode="track",
        filepath=DEFAULT_COCO_CONFIG,
        ignore_as_class=ignore_as_class,
    )

    out_paths: List[str] = []
    colors_list: List[List[np.ndarray]] = []
    poly2ds_list: List[List[List[Poly2D]]] = []

    logger.info("Preparing annotations for SegTrack to Bitmasks")

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_name = video_anns[0].video_name
        out_dir = os.path.join(out_base, video_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        for image_anns in video_anns:
            image_name = image_anns.name.replace(".jpg", ".png")
            image_name = os.path.split(image_name)[-1]
            out_path = os.path.join(out_dir, image_name)
            out_paths.append(out_path)

            colors: List[np.ndarray] = []
            poly2ds: List[List[Poly2D]] = []

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

                instance_id, global_instance_id = get_bdd100k_instance_id(
                    instance_id_maps, global_instance_id, str(label.id)
                )

                color = set_instance_color(
                    label, category_id, instance_id, category_ignored
                )
                colors.append(color)
                poly2ds.append(label.poly_2d)

            colors_list.append(colors)
            poly2ds_list.append(poly2ds)

    frames_to_masks(nproc, out_paths, colors_list, poly2ds_list)


def main() -> None:
    """Main function."""
    args = parse_args()
    assert args.mode in ["sem_seg", "ins_seg", "seg_track"]
    frames = start_converting(args)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # matplotlib offscreen render

    convert_func = dict(
        sem_seg=semseg_to_masks,
        ins_seg=insseg_to_bitmasks,
        seg_track=segtrack_to_bitmasks,
    )[args.mode]
    convert_func(
        frames,
        args.output,
        args.ignore_as_class,
        args.remove_ignore,
        args.nproc,
    )

    logger.info("Finished!")


if __name__ == "__main__":
    main()
