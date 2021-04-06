"""Common type definitions."""

import sys
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

# Used for Json loading
DictAny = Dict[str, Any]  # type: ignore[misc]
ListAny = List[Any]  # type: ignore[misc]


class CatType(TypedDict):
    """Define types of categories in GT."""

    supercategory: str
    id: int
    name: str


class AnnType(TypedDict, total=False):
    """Define types of annotations in GT."""

    id: int
    image_id: int
    category_id: int
    bdd100k_id: str
    iscrowd: int
    ignore: int
    instance_id: Optional[int]
    bbox: Optional[List[float]]
    area: Optional[float]
    segmentation: Optional[List[List[float]]]


class ImgType(TypedDict, total=False):
    """Define types of images in GT."""

    id: int
    file_name: str
    height: int
    width: int
    video_id: Optional[int]
    frame_id: Optional[int]


class VidType(TypedDict):
    """Define types of videos in GT."""

    id: int
    name: str


class GtType(TypedDict, total=False):
    """Define types of the GT in COCO format."""

    categories: List[CatType]
    annotations: List[AnnType]
    images: List[ImgType]
    type: str
    videos: Optional[List[VidType]]


class PredType(TypedDict):
    """Define input prediction type."""

    category: str
    score: float
    name: str
    bbox: List[float]
    image_id: int
    category_id: int
