"""Util functions."""

import os
import os.path as osp
from itertools import groupby
from typing import Dict, List, Tuple

from scalabel.common.io import load_config
from scalabel.label.to_coco import get_instance_id
from scalabel.label.typing import Label
from scalabel.label.utils import check_crowd, check_ignored

from .typing import BDD100KConfig


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


def check_bdd100k_crowd(label: Label) -> bool:
    """Check crowd attribute for BDD100K."""
    if label.id == "-1":
        return True
    return check_crowd(label)


def check_bdd100k_ignored(label: Label) -> bool:
    """Check ignored attribute for BDD100K."""
    if label.id == "-1":
        return True
    return check_ignored(label)


def load_bdd100k_config(cfg_path: str) -> BDD100KConfig:
    """Load a task-specific config."""
    if not cfg_path.endswith("toml"):
        cfg_path = osp.join(
            osp.split(osp.dirname(osp.abspath(__file__)))[0],
            "configs",
            cfg_path + ".toml",
        )
    config = load_config(cfg_path)
    return BDD100KConfig(**config)
