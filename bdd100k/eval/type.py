"""Define the types of the annotation files."""
import sys

if sys.version_info >= (3, 8):
    from typing import List, TypedDict
else:
    from typing import List
    from typing_extensions import TypedDict


class CatType(TypedDict):
    """Define types of categories in GT."""

    # pylint: disable=C0103
    supercategory: str
    id: int
    name: str


class AnnType(TypedDict):
    """Define types of annotations in GT."""

    is_crowd: int
    image_id: int
    bbox: List[float]
    area: float
    category_id: int
    ignore: int
    bdd100k_id: int
    segmentation: List[List[float]]


class ImgType(TypedDict):
    """Define types of images in GT."""

    # pylint: disable=C0103
    file_name: str
    height: int
    width: int
    id: int


class GtType(TypedDict):
    """Define types of the GT in COCO format."""

    categories: List[CatType]
    annotations: List[AnnType]
    images: List[ImgType]
    type: str


class PredType(TypedDict):
    """Define input prediction type."""

    category: str
    score: float
    name: str
    bbox: List[float]
    image_id: int
    category_id: int
