"""Common type definitions.

The annotation files in BDD100K format has additional annotations
('other person', 'other vehicle' and 'trail') besides the considered
categories ('car', 'pedestrian', 'truck', etc.) to indicate the uncertain
regions. Given the different handlings of these additional classes, we
provide three options to process the labels when converting them into COCO
format.
1. Ignore the labels. This is the default setting and is often used for
evaluation. CocoAPIs have native support for ignored annotations.
2. Remove the annotations from the label file. By setting
`remove-ignored=True`, the script will remove all the ignored annotations.
3. Use `ignored` as a separate class and the user can decide how to utilize
the annotations in `ignored` class. To achieve this, setting
`ignored-as-class=True`.
"""

import sys
from typing import Dict, List, Optional

from pydantic import BaseModel
from scalabel.common.typing import NDArrayU8
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
    ignored: bool
    mask: NDArrayU8
    bbox: List[float]
    area: float


class BDD100KConfig(BaseModel):
    """Extend metadata for BDD100K."""

    scalabel: Config
    remove_ignored: bool = False
    ignored_as_class: bool = False
    ignored_mapping: Optional[Dict[str, str]]
    name_mapping: Optional[Dict[str, str]]
