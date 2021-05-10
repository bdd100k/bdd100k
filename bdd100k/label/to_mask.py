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
from typing import Callable, Dict, List

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from PIL import Image
from scalabel.label.io import group_and_sort, load_label_config
from scalabel.label.to_coco import process_category
from scalabel.label.transforms import poly_to_patch
from scalabel.label.typing import Frame, Label, Poly2D
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import DEFAULT_LABEL_CONFIG, get_bdd100k_instance_id
from .label import drivables, labels
from .to_coco import parse_args, start_converting

IGNORE_LABEL = 255
SHAPE = np.array([720, 1280])
LANE_DIRECTION_MAP = {"parallel": 0, "vertical": 1}
LANE_STYLE_MAP = {"solid": 0, "dashed": 1}


def frame_to_mask(
    out_path: str,
    colors: List[np.ndarray],
    poly2ds: List[List[Poly2D]],
    with_instances: bool = True,
    back_color: int = 0,
    closed: bool = True,
) -> None:
    """Converting a frame of poly2ds to mask/bitmask."""
    assert len(colors) == len(poly2ds)

    if with_instances:
        img = np.ones([*SHAPE, 4], dtype=np.uint8) * back_color
    else:
        img = np.ones([*SHAPE, 1], dtype=np.uint8) * back_color

    if len(colors) == 0:
        pil_img = Image.fromarray(img.squeeze())
        pil_img.save(out_path)

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
                    # (0, 0, 0) for the background
                    color=(
                        ((i + 1) >> 8) / 255.0,
                        ((i + 1) % 255) / 255.0,
                        0.0,
                    ),
                    closed=closed,
                )
            )

    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    out = out.reshape((*SHAPE, -1)).astype(np.int32)
    out = (out[..., 0] << 8) + out[..., 1]
    plt.close()

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


def set_lane_color(label: Label, category_id: int) -> np.ndarray:
    """Set the color for the lane given its attributes and category."""
    attributes = label.attributes
    if attributes is None:
        lane_direction, lane_style = 0, 0
    else:
        lane_direction = LANE_DIRECTION_MAP[
            str(attributes.get("laneDirection", "parallel"))
        ]
        lane_style = LANE_STYLE_MAP[str(attributes.get("laneStyle", "solid"))]

    value = category_id + (lane_direction << 5) + (lane_style << 4)
    color = np.array([value], dtype=np.uint8)
    return color


def frames_to_masks(
    nproc: int,
    out_paths: List[str],
    colors_list: List[List[np.ndarray]],
    poly2ds_list: List[List[List[Poly2D]]],
    with_instances: bool = True,
    back_color: int = 0,
    closed: bool = True,
) -> None:
    """Execute the mask conversion in parallel."""
    with Pool(nproc) as pool:
        pool.starmap(
            partial(
                frame_to_mask,
                with_instances=with_instances,
                closed=closed,
                back_color=back_color,
            ),
            tqdm(
                zip(out_paths, colors_list, poly2ds_list),
                total=len(out_paths),
            ),
        )


def seg_to_masks(
    frames: List[Frame],
    out_base: str,
    ignore_as_class: bool = False,  # pylint: disable=unused-argument
    remove_ignore: bool = False,  # pylint: disable=unused-argument
    nproc: int = 4,
    mode: str = "sem_seg",
    back_color: int = IGNORE_LABEL,
    closed: bool = True,
) -> None:
    """Converting segmentation poly2d to 1-channel masks."""
    os.makedirs(out_base, exist_ok=True)

    out_paths: List[str] = []
    colors_list: List[List[np.ndarray]] = []
    poly2ds_list: List[List[List[Poly2D]]] = []

    categories = dict(sem_seg=labels, drivable=drivables)[mode]
    cat_name2id = {
        cat.name: cat.trainId for cat in categories if cat.trainId != 255
    }

    logger.info("Preparing annotations for Semseg to Bitmasks")

    for image_anns in tqdm(frames):
        image_name = image_anns.name.replace(".jpg", ".png")
        image_name = os.path.split(image_name)[-1]
        out_path = os.path.join(out_base, image_name)
        out_paths.append(out_path)

        colors: List[np.ndarray] = []
        poly2ds: List[List[Poly2D]] = []
        colors_list.append(colors)
        poly2ds_list.append(poly2ds)

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.category not in cat_name2id:
                continue
            if label.poly2d is None:
                continue

            category_id = cat_name2id[label.category]
            if mode in ["sem_seg", "drivable"]:
                color = np.array([category_id])
            else:
                color = set_lane_color(label, category_id)
            colors.append(color)
            poly2ds.append(label.poly2d)

    logger.info("Start Conversion for Seg to Masks")
    frames_to_masks(
        nproc,
        out_paths,
        colors_list,
        poly2ds_list,
        with_instances=False,
        back_color=back_color,
        closed=closed,
    )


ToMasksFunc = Callable[[List[Frame], str, bool, bool, int], None]
semseg_to_masks: ToMasksFunc = partial(
    seg_to_masks, mode="sem_seg", back_color=IGNORE_LABEL, closed=True
)
drivable_to_masks: ToMasksFunc = partial(
    seg_to_masks,
    mode="drivable",
    back_color=len(drivables) - 1,
    closed=True,
)
lanemark_to_masks: ToMasksFunc = partial(
    seg_to_masks, mode="lane_mark", back_color=IGNORE_LABEL, closed=False
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

    _, categories, name_mapping, ignore_mapping = load_label_config(
        filepath=DEFAULT_LABEL_CONFIG,
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
        colors_list.append(colors)
        poly2ds_list.append(poly2ds)

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.poly2d is None:
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
            poly2ds.append(label.poly2d)

    logger.info("Start conversion for InsSeg to Bitmasks")
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
    _, categories, name_mapping, ignore_mapping = load_label_config(
        filepath=DEFAULT_LABEL_CONFIG,
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
            colors_list.append(colors)
            poly2ds_list.append(poly2ds)

            for label in image_anns.labels:
                if label.poly2d is None:
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
                poly2ds.append(label.poly2d)

    logger.info("Start Conversion for SegTrack to Bitmasks")
    frames_to_masks(nproc, out_paths, colors_list, poly2ds_list)


def main() -> None:
    """Main function."""
    args = parse_args()
    assert args.mode in [
        "sem_seg",
        "drivable",
        "lane_mark",
        "ins_seg",
        "seg_track",
    ]
    frames = start_converting(args)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # matplotlib offscreen render

    convert_funcs: Dict[str, ToMasksFunc] = dict(
        sem_seg=semseg_to_masks,
        drivable=drivable_to_masks,
        lane_mark=lanemark_to_masks,
        ins_seg=insseg_to_bitmasks,
        seg_track=segtrack_to_bitmasks,
    )
    convert_funcs[args.mode](
        frames,
        args.output,
        args.ignore_as_class,
        args.remove_ignore,
        args.nproc,
    )

    logger.info("Finished!")


if __name__ == "__main__":
    main()
