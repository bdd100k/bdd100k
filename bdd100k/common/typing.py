"""Common type definitions."""

import sys
from typing import Dict, List, Optional

import numpy as np
from scalabel.label.typing import Config

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


class InstanceType(TypedDict, total=False):
    """Define types of annotations in GT."""

    instance_id: int
    category_id: int
    truncated: bool
    occluded: bool
    crowd: bool
    ignore: bool
    mask: np.ndarray
    bbox: List[float]
    area: float


class BDDConfig(Config):  # pylint: disable=too-few-public-methods
    """Extend metadata for BDD100K."""

    remove_ignore: bool = False
    ignore_as_class: bool = False
    ignore_mapping: Optional[Dict[str, str]]
    name_mapping: Optional[Dict[str, str]]
