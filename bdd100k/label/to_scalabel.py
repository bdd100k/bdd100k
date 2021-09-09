"""Convert BDD100K to Scalabel format."""

from typing import Dict, List, Optional

from scalabel.label.typing import Frame, Label
from scalabel.label.utils import get_leaf_categories
from tqdm import tqdm

from ..common.typing import BDD100KConfig

IGNORED = "ignored"


def deal_bdd100k_category(
    label: Label, bdd100k_config: BDD100KConfig, cat_name2id: Dict[str, int]
) -> Optional[Label]:
    """Deal with BDD100K category."""
    category_name = label.category
    if (
        bdd100k_config.name_mapping is not None
        and category_name in bdd100k_config.name_mapping
    ):
        category_name = bdd100k_config.name_mapping[category_name]

    if category_name not in cat_name2id:
        if bdd100k_config.remove_ignored:
            result = None
        elif bdd100k_config.ignored_as_class:
            assert IGNORED in cat_name2id
            category_name = IGNORED
            label.category = category_name
            result = label
        else:
            assert bdd100k_config.ignored_mapping is not None
            assert category_name in bdd100k_config.ignored_mapping
            category_name = bdd100k_config.ignored_mapping[category_name]
            if label.attributes is None:
                label.attributes = {}
            label.attributes[IGNORED] = True
            label.category = category_name
            result = label
    else:
        label.category = category_name
        result = label
    return result


def bdd100k_to_scalabel(
    frames: List[Frame], bdd100k_config: BDD100KConfig
) -> List[Frame]:
    """Converting BDD100K to Scalabel format."""
    categories = get_leaf_categories(bdd100k_config.scalabel.categories)
    cat_name2id = {cat.name: i + 1 for i, cat in enumerate(categories)}
    for image_anns in tqdm(frames):
        if image_anns.labels is not None:
            for i in reversed(range(len(image_anns.labels))):
                label = deal_bdd100k_category(
                    image_anns.labels[i], bdd100k_config, cat_name2id
                )
                if label is None:
                    image_anns.labels.pop(i)

    return frames
