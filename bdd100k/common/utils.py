"""Util functions."""

import os
import os.path as osp
from itertools import groupby
from typing import Dict, List, Tuple

from scalabel.label.typing import Label
from scalabel.label.to_coco import get_instance_id, get_object_attributes

DEFAULT_COCO_CONFIG = osp.join(
    osp.dirname(osp.abspath(__file__)), "configs.toml"
)


def list_files(
    inputs: str, suffix: str = "", with_prefix: bool = False
) -> List[str]:
    """List files paths for a folder/nested folder."""
    files: List[str] = []
    for root, _, file_iter in os.walk(inputs, topdown=True):
        path = osp.normpath(osp.relpath(root, inputs))
        path = "" if path == "." else path
        if with_prefix:
            path = osp.join(inputs, path)
        files.extend(
            [
                osp.join(path, file_)
                for file_ in file_iter
                if file_.endswith(suffix)
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


def get_bdd100k_instance_id(
    instance_id_maps: Dict[str, int], global_instance_id: int, scalabel_id: str
) -> Tuple[int, int]:
    """Get instance id given its corresponding Scalabel id for BDD100K."""
    if scalabel_id == "-1":
        instance_id = global_instance_id
        global_instance_id += 1
        return instance_id, global_instance_id
    return get_instance_id(instance_id_maps, global_instance_id, scalabel_id)


def get_bdd100k_object_attributes(
    label: Label, ignore: bool
) -> Tuple[int, int]:
    """Set attributes for the ann dict in BDD100K."""
    if label.id == "-1":
        ignore = True
    return get_object_attributes(label, ignore)
