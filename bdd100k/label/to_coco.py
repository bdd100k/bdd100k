"""Convert BDD100K to COCO format."""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayI32
from scalabel.label.coco_typing import AnnType, GtType, ImgType, VidType
from scalabel.label.io import group_and_sort, load
from scalabel.label.to_coco import (
    scalabel2coco_box_track,
    scalabel2coco_detection,
    scalabel2coco_ins_seg,
    scalabel2coco_pose,
    scalabel2coco_seg_track,
    set_seg_object_geometry,
)
from scalabel.label.transforms import get_coco_categories, mask_to_bbox
from scalabel.label.typing import Config, Frame, ImageSize
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
)
from tqdm import tqdm

from ..common.logger import logger
from ..common.typing import BDD100KConfig, InstanceType
from ..common.utils import (
    get_bdd100k_instance_id,
    group_and_sort_files,
    list_files,
    load_bdd100k_config,
)
from .to_scalabel import bdd100k_to_scalabel


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="bdd100k to coco format")
    parser.add_argument(
        "-i", "--input", required=True, help="path to Scalabel label file"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="path to save coco formatted label file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=[
            "det",
            "ins_seg",
            "box_track",
            "seg_track",
            "pose",
        ],
        help="conversion mode",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration for COCO categories",
    )
    parser.add_argument(
        "-mb",
        "--mask-base",
        type=str,
        default=None,
        help="Path to the BitMasks base folder.",
    )
    parser.add_argument(
        "-om",
        "--only-mask",
        action="store_true",
        help="Convert only masks.",
    )
    return parser.parse_args()


def bitmasks_loader(mask_name: str) -> Tuple[List[InstanceType], ImageSize]:
    """Parse instances from the bitmask."""
    if mask_name.endswith(".jpg"):
        mask_name = mask_name.replace(".jpg", ".png")
    bitmask: NDArrayI32 = np.asarray(Image.open(mask_name), dtype=np.int32)
    category_map = bitmask[:, :, 0]
    attributes_map = bitmask[:, :, 1]
    instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]
    indentity_map = (
        (category_map << 24) + (attributes_map << 16) + instance_map
    )

    instances: List[InstanceType] = []

    identities: NDArrayI32 = np.unique(indentity_map)
    for identity in identities:
        mask = np.equal(indentity_map, identity)
        category_id = (identity >> 24) & 255
        attribute = (identity >> 16) & 255
        instance_id = identity & 65535
        if category_id == 0:
            continue

        bbox = mask_to_bbox(mask)
        area = np.sum(mask).tolist()

        instance = InstanceType(
            instance_id=int(instance_id),
            category_id=int(category_id),
            truncated=bool(attribute & (1 << 3)),
            occluded=bool(attribute & (1 << 2)),
            crowd=bool(attribute & (1 << 1)),
            ignored=bool(attribute & (1 << 0)),
            mask=mask,
            bbox=bbox,
            area=area,
        )
        instances.append(instance)

    instances = sorted(instances, key=lambda instance: instance["instance_id"])
    img_shape = ImageSize(height=bitmask.shape[0], width=bitmask.shape[1])

    return (instances, img_shape)


def bitmask2coco_wo_ids(image: ImgType, mask_base: str) -> List[AnnType]:
    """Convert bitmasks annotations of an image to RLEs or polygons."""
    mask_name = os.path.join(mask_base, image["file_name"])
    instances, img_shape = bitmasks_loader(mask_name)
    image["height"] = img_shape.height
    image["width"] = img_shape.width

    annotations: List[AnnType] = []
    for instance in instances:
        annotation = AnnType(
            id=0,  # set further
            image_id=image["id"],
            category_id=instance["category_id"],
            instance_id=instance["instance_id"],
            iscrowd=instance["crowd"],
        )
        annotation = set_seg_object_geometry(annotation, instance["mask"])
        annotations.append(annotation)
    return annotations


def bitmask2coco_wo_ids_parallel(
    mask_base: str, images: List[ImgType], nproc: int = NPROC
) -> List[AnnType]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        annotations_list = pool.map(
            partial(bitmask2coco_wo_ids, mask_base=mask_base),
            tqdm(images),
        )
    annotations: List[AnnType] = []
    for anns in annotations_list:
        annotations.extend(anns)

    annotations = sorted(annotations, key=lambda ann: ann["image_id"])
    for i, annotation in enumerate(annotations):
        ann_id = i + 1
        annotation["id"] = ann_id
    return annotations


def bitmask2coco_with_ids(
    annotations: List[AnnType],
    mask_name: str,
    category_ids: List[int],
    instance_ids: List[int],
) -> List[AnnType]:
    """Convert bitmasks annotations of an image to RLEs or polygons."""
    bitmask: NDArrayI32 = np.asarray(Image.open(mask_name), dtype=np.int32)
    category_map = bitmask[..., 0]
    instance_map = (bitmask[..., 2] << 2) + bitmask[..., 3]
    for annotation, category_id, instance_id in zip(
        annotations, category_ids, instance_ids
    ):
        mask = np.logical_and(
            category_map == category_id, instance_map == instance_id
        )
        annotation = set_seg_object_geometry(annotation, mask)
    annotations = [
        ann for ann in annotations if "bbox" in ann and "segmentation" in ann
    ]
    return annotations


def bitmask2coco_with_ids_parallel(
    annotations_list: List[List[AnnType]],
    mask_names: List[str],
    category_ids_list: List[List[int]],
    instance_ids_list: List[List[int]],
    nproc: int = NPROC,
) -> List[AnnType]:
    """Execute the bitmask conversion in parallel."""
    logger.info("Converting annotations...")

    with Pool(nproc) as pool:
        annotations_list = pool.starmap(
            bitmask2coco_with_ids,
            tqdm(
                zip(
                    annotations_list,
                    mask_names,
                    category_ids_list,
                    instance_ids_list,
                ),
                total=len(annotations_list),
            ),
        )
    annotations: List[AnnType] = []
    for anns in annotations_list:
        annotations.extend(anns)

    return annotations


def bitmask2coco_ins_seg(
    mask_base: str, config: Config, nproc: int = NPROC
) -> GtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    files = list_files(mask_base, suffix=".png")
    images: List[ImgType] = []

    logger.info("Collecting bitmasks...")

    image_id = 0
    for file_ in tqdm(files):
        image_id += 1
        image = ImgType(
            id=image_id,
            file_name=file_.replace(".png", ".jpg"),
        )
        images.append(image)

    annotations = bitmask2coco_wo_ids_parallel(mask_base, images, nproc)
    return GtType(
        type="instances",
        categories=get_coco_categories(config),
        images=images,
        annotations=annotations,
    )


def bitmask2coco_seg_track(
    mask_base: str, config: Config, nproc: int = NPROC
) -> GtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    videos: List[VidType] = []
    images: List[ImgType] = []
    all_files = list_files(mask_base, suffix=".png")
    files_list = group_and_sort_files(all_files)

    logger.info("Collecting bitmasks...")

    video_id, image_id = 0, 0
    for files in files_list:
        video_name = os.path.split(files[0])[0]
        video_id += 1
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for frame_id, file_ in tqdm(enumerate(files)):
            image_id += 1
            image = ImgType(
                video_id=video_id,
                frame_id=frame_id,
                id=image_id,
                file_name=file_.replace(".png", ".jpg"),
            )
            images.append(image)

    annotations = bitmask2coco_wo_ids_parallel(mask_base, images, nproc)
    return GtType(
        type="instances",
        categories=get_coco_categories(config),
        videos=videos,
        images=images,
        annotations=annotations,
    )


def bdd100k2coco_ins_seg(
    mask_base: str, frames: List[Frame], config: Config, nproc: int = NPROC
) -> GtType:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    image_id, ann_id = 0, 0
    img_shape = config.imageSize
    images: List[ImgType] = []

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[AnnType]] = []

    categories = get_leaf_categories(config.categories)
    cat_name2id = {cat.name: i + 1 for i, cat in enumerate(categories)}

    logger.info("Collecting annotations...")

    for image_anns in tqdm(frames):
        image_id += 1
        if img_shape is None:
            if image_anns.size is not None:
                img_shape = image_anns.size
            else:
                raise ValueError("Image shape not defined!")

        image = ImgType(
            id=image_id,
            file_name=image_anns.name,
            height=img_shape.height,
            width=img_shape.width,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

        mask_name = os.path.join(
            mask_base,
            # Bitmask in .png format, image in .jpg format
            image_anns.name.replace(".jpg", ".png"),
        )
        mask_names.append(mask_name)

        category_ids: List[int] = []
        instance_ids: List[int] = []
        annotations: List[AnnType] = []

        instance_id = 0
        for label in image_anns.labels:
            if label.poly2d is None:
                continue
            if label.category not in cat_name2id:
                continue

            ann_id += 1
            instance_id += 1
            category_id = cat_name2id[label.category]
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=int(check_crowd(label) or check_ignored(label)),
                ignore=0,
            )

            category_ids.append(category_id)
            instance_ids.append(instance_id)
            annotations.append(annotation)

        category_ids_list.append(category_ids)
        instance_ids_list.append(instance_ids)
        annotations_list.append(annotations)

    annotations = bitmask2coco_with_ids_parallel(
        annotations_list,
        mask_names,
        category_ids_list,
        instance_ids_list,
        nproc,
    )

    return GtType(
        type="instances",
        categories=get_coco_categories(config),
        images=images,
        annotations=annotations,
    )


def bdd100k2coco_seg_track(
    mask_base: str, frames: List[Frame], config: Config, nproc: int = NPROC
) -> GtType:
    """Converting BDD100K Segmentation Tracking Set to COCO format."""
    video_id, image_id, ann_id = 0, 0, 0
    img_shape = config.imageSize
    frames_list = group_and_sort(frames)
    videos: List[VidType] = []
    images: List[ImgType] = []

    mask_names: List[str] = []
    category_ids_list: List[List[int]] = []
    instance_ids_list: List[List[int]] = []
    annotations_list: List[List[AnnType]] = []

    categories = get_leaf_categories(config.categories)
    cat_name2id = {cat.name: i + 1 for i, cat in enumerate(categories)}

    logger.info("Collecting annotations...")

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = {}

        video_name = video_anns[0].videoName
        video_id += 1
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in video_anns:
            image_id += 1
            if img_shape is None:
                if image_anns.size is not None:
                    img_shape = image_anns.size
                else:
                    raise ValueError("Image shape not defined!")

            image = ImgType(
                video_id=video_id,
                frame_id=image_anns.frameIndex,
                id=image_id,
                file_name=os.path.join(video_name, image_anns.name),
                height=img_shape.height,
                width=img_shape.width,
            )
            if image_anns.url is not None:
                image["coco_url"] = image_anns.url
            images.append(image)

            mask_name = os.path.join(
                mask_base,
                video_name,
                # Bitmask in .png format, image in .jpg format
                image_anns.name.replace(".jpg", ".png"),
            )
            mask_names.append(mask_name)

            category_ids: List[int] = []
            instance_ids: List[int] = []
            annotations: List[AnnType] = []

            for label in image_anns.labels:
                if label.poly2d is None:
                    continue
                if label.category not in cat_name2id:
                    continue

                ann_id += 1
                instance_id, global_instance_id = get_bdd100k_instance_id(
                    instance_id_maps, global_instance_id, label.id
                )
                category_id = cat_name2id[label.category]
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    instance_id=instance_id,
                    category_id=category_id,
                    scalabel_id=label.id,
                    iscrowd=int(check_crowd(label) or check_ignored(label)),
                    ignore=0,
                )

                category_ids.append(category_id)
                instance_ids.append(instance_id)
                annotations.append(annotation)

            category_ids_list.append(category_ids)
            instance_ids_list.append(instance_ids)
            annotations_list.append(annotations)

    annotations = bitmask2coco_with_ids_parallel(
        annotations_list,
        mask_names,
        category_ids_list,
        instance_ids_list,
        nproc,
    )

    return GtType(
        type="instances",
        categories=get_coco_categories(config),
        videos=videos,
        images=images,
        annotations=annotations,
    )


def main() -> None:
    """Main function."""
    args = parse_args()

    if args.only_mask:
        assert args.mode in ["ins_seg", "seg_track"]
        convert_function = dict(
            ins_seg=bitmask2coco_ins_seg,
            seg_track=bitmask2coco_seg_track,
        )[args.mode]

        cfg_path = args.config if args.config is not None else args.mode
        bdd100k_config = load_bdd100k_config(cfg_path)
        logger.info("Start format converting...")
        coco = convert_function(
            args.input, bdd100k_config.scalabel, args.nproc
        )
    else:
        logger.info("Loading annotations...")
        dataset = load(args.input, args.nproc)
        if args.config is not None:
            bdd100k_config = load_bdd100k_config(args.config)
        elif dataset.config is not None:
            bdd100k_config = BDD100KConfig(config=dataset.config)
        else:
            bdd100k_config = load_bdd100k_config(args.mode)

        if args.mode in ["det", "box_track", "pose"]:
            convert_func = dict(
                det=scalabel2coco_detection,
                box_track=scalabel2coco_box_track,
                pose=scalabel2coco_pose,
            )[args.mode]
        else:
            if args.mask_base is not None:
                convert_func = partial(
                    dict(
                        ins_seg=bdd100k2coco_ins_seg,
                        seg_track=bdd100k2coco_seg_track,
                    )[args.mode],
                    mask_base=args.mask_base,
                    nproc=args.nproc,
                )
            else:
                convert_func = partial(
                    dict(
                        ins_seg=scalabel2coco_ins_seg,
                        seg_track=scalabel2coco_seg_track,
                    )[args.mode],
                    nproc=args.nproc,
                )

        logger.info("Start format converting...")
        frames = bdd100k_to_scalabel(dataset.frames, bdd100k_config)
        coco = convert_func(frames=frames, config=bdd100k_config.scalabel)

    logger.info("Saving converted annotations to disk...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
