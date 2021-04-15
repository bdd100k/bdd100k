"""Util functions."""

import glob
import os
import os.path as osp
from itertools import groupby
from typing import List

from scalabel.label.io import load as load_bdd100k
from scalabel.label.typing import Frame

DEFAULT_COCO_CONFIG = osp.join(
    osp.dirname(osp.abspath(__file__)), "configs.toml"
)


def list_files(inputs: str, suffix: str = "") -> List[str]:
    """List files paths for a folder/nested folder."""
    files: List[str] = []
    for root, _, files in os.walk(inputs, topdown=True):
        files.extend(
            [
                osp.join(root, file_)
                for file_ in files
                if osp.splitext(file_)[1] == suffix
            ]
        )
    files = sorted(files)
    return files


def group_and_sort_files(files: List[str]) -> List[List[str]]:
    """Group frames by video_name and sort."""
    files_list: List[List[str]] = []
    for _, files_iter in groupby(files, lambda file_: osp.split(file_)[0]):
        files_list.append(sorted(list(files_iter)))
    files_list = sorted(files_list, key=lambda files: files[0])
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

    outputs = sorted(outputs, key=lambda output: output.name)
    return outputs
