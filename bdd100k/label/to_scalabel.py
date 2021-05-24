"""Convert BDD100K to COCO format."""

from typing import Dict, List

from scalabel.label.typing import Frame, Label
from scalabel.label.utils import get_leaf_categories
from tqdm import tqdm

from ..common.typing import BDD100KConfig

IGNORED = "ignored"


def deal_bdd100k_category(
    label: Label, bdd100k_config: BDD100KConfig, cat_name2id: Dict[str, int]
) -> Label:
    """Deal with BDD100K category."""
    category_name = label.category
    if (
        bdd100k_config.name_mapping is not None
        and category_name in bdd100k_config.name_mapping
    ):
        category_name = bdd100k_config.name_mapping[category_name]

    if category_name not in cat_name2id:
        if bdd100k_config.remove_ignore:
            if label.attributes is None:
                label.attributes = dict()
            label.attributes[IGNORED] = True
        elif bdd100k_config.ignore_as_class:
            assert IGNORED in cat_name2id
            category_name = IGNORED
        else:
            assert bdd100k_config.ignore_mapping is not None
            assert category_name in bdd100k_config.ignore_mapping
            category_name = bdd100k_config.ignore_mapping[category_name]
    label.category = category_name
    return label


def bdd100k_to_scalabel(
    frames: List[Frame], bdd100k_config: BDD100KConfig
) -> List[Frame]:
    """Converting BDD100K Instance Segmentation Set to COCO format."""
    categories = get_leaf_categories(bdd100k_config.config.categories)
    cat_name2id = {cat.name: i + 1 for i, cat in enumerate(categories)}
    for image_anns in tqdm(frames):
        for label in image_anns.labels:
            label = deal_bdd100k_category(label, bdd100k_config, cat_name2id)
    return frames
