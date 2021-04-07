"""Util functions."""

import glob
import os
import os.path as osp
from typing import Dict, List, Tuple

import toml
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


def load_categories(
    mode: str = "det",
    ignore_as_class: bool = False,
) -> Tuple[List[CatType], Dict[str, str], Dict[str, str]]:
    """Load the annotation dictionary."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_name = "det" if mode == "det" else "other"
    cfg_file = "{}/{}.toml".format(cur_dir, cfg_name)
    categories: List[CatType] = toml.load(cfg_file)["categories"]

    if ignore_as_class:
        categories.append(
            CatType(
                supercategory="none", id=len(categories) + 1, name="ignored"
            )
        )
    name_mapping = NAME_MAPPING.copy()
    ignore_mapping = IGNORE_MAP.copy()

    return categories, name_mapping, ignore_mapping


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
