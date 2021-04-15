"""Util functions."""

import glob
import os
import os.path as osp
from itertools import groupby
from typing import List

import numpy as np
from PIL import Image
from scalabel.label.io import load as load_bdd100k
from scalabel.label.typing import Frame

from ..common.typing import InstanceType

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


def bitmasks_loader(mask_name: str) -> List[InstanceType]:
    """Parse instances from the bitmask."""
    bitmask = np.asarray(Image.open(mask_name)).astype(np.int32)
    category_map = bitmask[:, :, 0]
    attributes_map = bitmask[:, :, 1]
    instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

    instances: List[InstanceType] = []

    # 0 is for the background
    instance_ids = np.sort(np.unique(instance_map[instance_map >= 1]))
    for instance_id in instance_ids:
        mask_inds_i = instance_map == instance_id
        attributes_i = np.unique(attributes_map[mask_inds_i])
        category_ids_i = np.unique(category_map[mask_inds_i])

        assert attributes_i.shape[0] == 1
        assert category_ids_i.shape[0] == 1
        attribute = attributes_i[0]
        category_id = category_ids_i[0]

        instance = InstanceType(
            instance_id=instance_id,
            category_id=category_id,
            truncated=bool(attribute & (1 << 3)),
            occluded=bool(attribute & (1 << 2)),
            crowd=bool(attribute & (1 << 1)),
            ignore=bool(attribute & (1 << 1)),
            mask=mask_inds_i.astype(np.int32),
        )
        instances.append(instance)

    return instances
