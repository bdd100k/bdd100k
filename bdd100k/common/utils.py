"""Util functions."""

import os
import os.path as osp
from itertools import groupby
from typing import Dict, List, Tuple

from scalabel.common.io import load_config
from scalabel.label.to_coco import get_instance_id
from scalabel.label.typing import Label
from scalabel.label.utils import get_leaf_categories

from .typing import BDDConfig

DEFAULT_LABEL_CONFIG = osp.join(
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


def get_bdd100k_iscrowd(label: Label, ignore: bool) -> int:
    """Set attributes for the ann dict in BDD100K."""
    if label.id == "-1":
        return 1
    if label.attributes is None:
        return 0
    crowd = label.attributes.get("crowd", False)
    return int(crowd or ignore)


def load_bdd_config(filepath: str) -> BDDConfig:
    """Load label configuration from a config file (toml / yaml)."""
    cfg = load_config(filepath)
    config = BDDConfig(**cfg)
    return config


def get_bdd100k_category_id(
    category_name: str, config: BDDConfig
) -> Tuple[bool, int]:
    """Get category id from category name and MetaConfig.

    The returned boolean item means whether this instance should be ignored.
    """
    leaf_cats = get_leaf_categories(config.categories)
    leaf_cat_names = [cat.name for cat in leaf_cats]
    if config.ignore_as_class:
        leaf_cat_names.append("ignore")

    if category_name not in leaf_cat_names:
        if config.remove_ignore:
            return True, 0

        if config.ignore_as_class:
            category_name = "ignore"
        else:
            assert config.ignore_mapping is not None
            assert category_name in config.ignore_mapping
            category_name = config.ignore_mapping[category_name]
    category_id = leaf_cat_names.index(category_name) + 1
    return False, category_id
