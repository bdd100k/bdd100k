"""Type definition for scalabel format."""
import sys
from typing import Optional

import numpy as np

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


class ImgInfo(TypedDict, totoal=False):
    """Define types of the image information for the returned item."""

    filename: str
    height: Optional[int]
    width: Optional[int]


class AnnInfo(TypedDict, total=False):
    """Define types of the annotation information for the returned item."""

    bboxes: np.ndarray
    labels: np.ndarray
    bboxes_ignore: np.ndarray
    masks: Optional[np.ndarray]
    seg_map: Optional[str]
