"""Common type definitions."""

import sys
from typing import Any, Dict, List

import numpy as np

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


DictAny = Dict[str, Any]  # type: ignore[misc]
ListAny = List[Any]  # type: ignore[misc]


class InstanceType(TypedDict, total=False):
    """Define types of annotations in GT."""

    instance_id: int
    category_id: int
    truncated: bool
    occluded: bool
    crowd: bool
    ignore: bool
    mask: np.ndarray
