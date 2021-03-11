"""Convert BDD100K to COCO format.

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
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage import measure

from ..common.iterator import ImageLabelIterator, VideoLabelIterator
from ..common.logger import logger
from ..common.typing import DictAny, ListAny
from ..common.utils import read


def parser_definition() -> argparse.ArgumentParser:
    """Definition of the parser."""
    parser = argparse.ArgumentParser(description="BDD100K to COCO format")
    parser.add_argument(
        "-i",
        "--in-path",
        default="/input/path/",
        help="Path to detection JSON file or tracking base folder.",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        default="/output/path",
        help="Path to save coco formatted label file.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["det", "box_track", "seg_track"],
        help="conversion mode: detection or tracking.",
    )
    parser.add_argument(
        "-mb",
        "--mask-base",
        help="Path to the BitMasks base folder.",
    )
    parser.add_argument(
        "-ri",
        "--remove-ignore",
        action="store_true",
        help="Remove the ignored annotations from the label file.",
    )
    parser.add_argument(
        "-ic",
        "--ignore-as-class",
        action="store_true",
        help="Put the ignored annotations to the `ignored` category.",
    )
    return parser


def close_contour(contour: ListAny) -> ListAny:
    """Explicitly close the contour."""
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def mask_to_polygon(
    binary_mask: np.array, x_1: int, y_1: int, tolerance: int = 2
) -> List[ListAny]:
    """Convert BitMask to polygon."""
    polygons = []
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        for i, _ in enumerate(segmentation):
            if i % 2 == 0:
                segmentation[i] = (segmentation[i] + x_1).tolist()
            else:
                segmentation[i] = (segmentation[i] + y_1).tolist()

        polygons.append(segmentation)

    return polygons


def parse_box_object(
    labels_per_object: DictAny,
) -> Tuple[List[int], float, List[List[int]]]:
    """Parsing bbox, area, polygon from bbox object."""
    x1 = labels_per_object["box2d"]["x1"]
    y1 = labels_per_object["box2d"]["y1"]
    x2 = labels_per_object["box2d"]["x2"]
    y2 = labels_per_object["box2d"]["y2"]

    bbox = [x1, y1, x2 - x1, y2 - y1]
    area = float((x2 - x1) * (y2 - y1))
    polygon = [[x1, y1, x1, y2, x2, y2, x2, y1]]
    return bbox, area, polygon


def parse_seg_object(
    labels_per_object: DictAny,
) -> Tuple[List[int], float, List[List[int]]]:
    """Parsing bbox, area, polygon from seg object."""
    mask = np.logical_and(
        labels_per_object["category_map"] == labels_per_object["category_id"],
        labels_per_object["instance_map"] == labels_per_object["instance_id"],
    )
    if not mask.sum():
        return [], 0.0, [[]]

    x_inds = np.sum(mask, axis=0).nonzero()[0]
    y_inds = np.sum(mask, axis=1).nonzero()[0]
    x1, x2 = np.min(x_inds), np.max(x_inds) + 1
    y1, y2 = np.min(y_inds), np.max(y_inds) + 1
    bbox = np.array([x1, y1, x2 - x1, y2 - y1]).tolist()

    mask = mask[y1:y2, x1:x2]
    area = np.sum(mask).tolist()
    polygon = mask_to_polygon(mask, x1, y1)
    return bbox, area, polygon


class Det2COCOIterator(ImageLabelIterator):
    """Iterator for conversion of detection to coco format."""

    def __init__(
        self, ignore_as_class: bool = False, remove_ignore: bool = False
    ) -> None:
        """Initialize the det2cooc iterator."""
        super().__init__("det", ignore_as_class, remove_ignore)
        self.naming_replacement_dict = {
            "person": "pedestrian",
            "motor": "motorcycle",
            "bike": "bicycle",
        }
        self.parse_object = parse_box_object

    def image_iteration(self, labels_per_image: DictAny) -> None:
        """Actions for the video iteration."""
        super().image_iteration(labels_per_image)
        image = dict(
            file_name=labels_per_image["name"],
            height=720,
            width=1280,
            id=self.image_id,
        )
        self.coco["images"].append(image)

    def object_iteration(self, labels_per_object: DictAny) -> None:
        """Actions for the object iteration."""
        if labels_per_object["category"] in self.naming_replacement_dict:
            labels_per_object["category"] = self.naming_replacement_dict[
                labels_per_object["category"]
            ]
        super().object_iteration(labels_per_object)
        if labels_per_object["category_ignored"] and self.remove_ignore:
            return

        bbox, area, polygon = self.parse_object(labels_per_object)
        if area == 0:
            return

        ann = dict(
            id=self.object_id,
            image_id=self.image_id,
            category_id=labels_per_object["category_id"],
            bdd100k_id=labels_per_object["id"],
            occluded=int(
                labels_per_object["attributes"].get("Occluded", False)
            ),
            truncated=int(
                labels_per_object["attributes"].get("Truncated", False)
            ),
            iscrowd=int(labels_per_object["attributes"].get("Crowd", False))
            or int(labels_per_object["category_ignored"]),
            ignore=int(labels_per_object["category_ignored"]),
            bbox=bbox,
            area=area,
            segmentation=polygon,
        )
        self.coco["annotations"].append(ann)

    def after_iteration(self) -> DictAny:
        """Return the coco dict."""
        self.coco["type"] = "instances"
        return self.coco


class BoxTrack2COCOIterator(VideoLabelIterator):
    """Iterator for conversion of boxtrack to coco format."""

    def __init__(
        self, ignore_as_class: bool = False, remove_ignore: bool = False
    ) -> None:
        """Initialize the boxtrack2cooc iterator."""
        super().__init__("track", ignore_as_class, remove_ignore)
        self.parse_object = parse_box_object

    def video_iteration(self, labels_per_video: List[DictAny]) -> None:
        """Actions for the video iteration."""
        super().video_iteration(labels_per_video)
        assert len(labels_per_video) > 0
        video_name = labels_per_video[0]["video_name"]
        video = dict(id=self.video_id, name=video_name)
        self.coco["videos"].append(video)

    def image_iteration(self, labels_per_image: DictAny) -> None:
        """Actions for the image iteration."""
        super().image_iteration(labels_per_image)
        image = dict(
            file_name=os.path.join(
                labels_per_image["video_name"], labels_per_image["name"]
            ),
            height=720,
            width=1280,
            id=self.image_id,
            video_id=self.video_id,
            frame_id=labels_per_image["index"],
        )
        self.coco["images"].append(image)

    def object_iteration(self, labels_per_object: DictAny) -> None:
        """Actions for the object iteration."""
        super().object_iteration(labels_per_object)

        if labels_per_object["category_ignored"] and self.remove_ignore:
            return

        bbox, area, polygon = self.parse_object(labels_per_object)
        if area == 0:
            return

        ann = dict(
            id=self.object_id,
            image_id=self.image_id,
            category_id=labels_per_object["category_id"],
            instance_id=labels_per_object["instance_id"],
            bdd100k_id=labels_per_object["id"],
            occluded=int(
                labels_per_object["attributes"].get("Occluded", False)
            ),
            truncated=int(
                labels_per_object["attributes"].get("Truncated", False)
            ),
            iscrowd=int(labels_per_object["attributes"].get("Crowd", False))
            or int(labels_per_object["category_ignored"]),
            ignore=int(labels_per_object["category_ignored"]),
            bbox=bbox,
            area=area,
            segmentation=polygon,
        )
        self.coco["annotations"].append(ann)

    def after_iteration(self) -> DictAny:
        """Return the coco dict."""
        return self.coco


class SegTrack2COCOIterator(BoxTrack2COCOIterator):
    """Iterator for conversion of segtrack to coco format."""

    def __init__(
        self,
        mask_base: str,
        ignore_as_class: bool = False,
        remove_ignore: bool = False,
    ) -> None:
        """Initialize the segtrack2coco iterator."""
        super().__init__(ignore_as_class, remove_ignore)
        assert os.path.isdir(mask_base)
        self.mask_base = mask_base
        self.parse_object = parse_seg_object

    def image_iteration(self, labels_per_image: DictAny) -> None:
        """Actions for the image iteration."""
        super().image_iteration(labels_per_image)
        mask_name = labels_per_image["name"].replace(".jpg", ".png")
        mask_path = os.path.join(
            self.mask_base, labels_per_image["video_name"], mask_name
        )
        bitmask = np.asarray(Image.open(mask_path)).astype(np.int32)
        labels_per_image["category_map"] = bitmask[:, :, 0]
        labels_per_image["instance_map"] = (bitmask[:, :, 2] << 8) + bitmask[
            :, :, 3
        ]
        for labels_per_object in labels_per_image["labels"]:
            labels_per_object["category_map"] = labels_per_image[
                "category_map"
            ]
            labels_per_object["instance_map"] = labels_per_image[
                "instance_map"
            ]


def start_converting() -> Tuple[argparse.Namespace, List[List[DictAny]]]:
    """Parses arguments, and logs settings."""
    parser = parser_definition()
    args = parser.parse_args()

    logger.info(
        "Mode: %s\nremove-ignore: %s\nignore-as-class: %s",
        args.mode,
        args.remove_ignore,
        args.ignore_as_class,
    )
    logger.info("Loading annotations...")
    labels = read(args.in_path)

    logger.info("Converting annotations...")
    return args, labels


def main() -> None:
    """Main function."""
    args, labels = start_converting()
    out_fn = os.path.join(args.out_path)

    if args.mode == "det":
        iterator: ImageLabelIterator = Det2COCOIterator(
            args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "box_track":
        iterator = BoxTrack2COCOIterator(
            args.ignore_as_class, args.remove_ignore
        )
    elif args.mode == "seg_track":
        iterator = SegTrack2COCOIterator(
            args.mask_base, args.ignore_as_class, args.remove_ignore
        )
    coco = iterator(labels)

    logger.info("Saving converted annotations to disk...")
    with open(out_fn, "w") as f:
        json.dump(coco, f)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
