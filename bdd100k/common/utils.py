"""Util functions."""

import os
import os.path as osp
from itertools import groupby
from typing import Dict, List, Tuple

from scalabel.common.io import load_config
from scalabel.label.to_coco import get_instance_id
from scalabel.label.typing import Label
from scalabel.label.utils import check_crowd, check_ignored

from .logger import logger
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
    assert osp.exists(cfg_path), f"Task config {cfg_path} does not exist."
    config = load_config(cfg_path)
    return BDD100KConfig(**config)


def reorder_preds(gt_paths: List[str], pred_paths: List[str]) -> List[str]:
    """Reorder the order of predictions given groundtruths."""
    pred_map: Dict[str, str] = {
        osp.splitext(osp.split(pred_path)[-1])[0]: pred_path
        for pred_path in pred_paths
    }
    sorted_results: List[str] = []
    miss_num = 0
    for gt_path in gt_paths:
        gt_name = osp.splitext(osp.split(gt_path)[-1])[0]
        if gt_name in pred_map:
            sorted_results.append(pred_map[gt_name])
        else:
            sorted_results.append("")
            miss_num += 1
    logger.info("%s images are missed in the prediction.", miss_num)
    return sorted_results
