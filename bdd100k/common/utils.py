"""Util functions."""

import glob
import os
import os.path as osp
from itertools import groupby
from typing import Dict, List, Tuple

from scalabel.label.io import load as load_bdd100k
from scalabel.label.typing import Frame

from ..common.typing import CatType

NAME_MAPPING: Dict[str, str] = {
    "bike": "bicycle",
    "caravan": "car",
    "motor": "motorcycle",
    "person": "pedestrian",
    "van": "car",
}

IGNORE_MAP: Dict[str, str] = {
    "other person": "pedestrian",
    "other vehicle": "car",
    "trailer": "truck",
}

CATEGORIES: List[CatType] = [
    CatType(supercategory="human", id=1, name="pedestrian"),
    CatType(supercategory="human", id=2, name="rider"),
    CatType(supercategory="vehicle", id=3, name="car"),
    CatType(supercategory="vehicle", id=4, name="truck"),
    CatType(supercategory="vehicle", id=5, name="bus"),
    CatType(supercategory="vehicle", id=6, name="train"),
    CatType(supercategory="bike", id=7, name="motorcycle"),
    CatType(supercategory="bike", id=8, name="bicycle"),
]


def init(
    mode: str = "det",
    ignore_as_class: bool = False,
) -> Tuple[List[CatType], Dict[str, int]]:
    """Initialze the annotation dictionary."""
    categories = CATEGORIES.copy()
    if mode == "det":
        categories += [
            {
                "supercategory": "traffic light",
                "id": 9,
                "name": "traffic light",
            },
            {
                "supercategory": "traffic sign",
                "id": 10,
                "name": "traffic sign",
            },
        ]

    if ignore_as_class:
        categories.append(
            {
                "supercategory": "none",
                "id": len(categories) + 1,
                "name": "ignored",
            }
        )
    category_name2id: Dict[str, int] = {
        str(category["name"]): int(category["id"]) for category in categories
    }

    return categories, category_name2id


def list_files(inputs: str) -> List[List[str]]:
    """List files names for a folder/nested folder."""
    files_list: List[List[str]] = []
    assert osp.isdir(inputs)
    sub_dirs = sorted(os.listdir(inputs))
    for sub_dir in sub_dirs:
        dir_path = osp.join(inputs, sub_dir)
        assert osp.isdir(dir_path)
        files = sorted(os.listdir(dir_path))
        files = [osp.join(dir_path, file_name) for file_name in files]
        files_list.append(files)
    return files_list


def read(inputs: str) -> List[Frame]:
    """Read annotations from file/files."""
    outputs: List[Frame] = []
    if osp.isdir(inputs):
        files = glob.glob(osp.join(inputs, "*.json"))
        for file_ in files:
            outputs.extend(load_bdd100k(file_))
    elif osp.isfile(inputs) and inputs.endswith("json"):
        outputs.extend(load_bdd100k(inputs))
    else:
        raise TypeError("Inputs must be a folder or a JSON file.")

    return outputs


def group_and_sort(inputs: List[Frame]) -> List[List[Frame]]:
    """Group frames by video_name and sort."""
    for frame in inputs:
        assert frame.video_name is not None
        assert frame.frame_index is not None
    frames_list: List[List[Frame]] = []
    for _, frame_iter in groupby(inputs, lambda frame: frame.video_name):
        frames = sorted(
            list(frame_iter),
            key=lambda frame: frame.frame_index if frame.frame_index else 0,
        )
        frames_list.append(frames)
    frames_list = sorted(
        frames_list, key=lambda frames: str(frames[0].video_name)
    )
    return frames_list
