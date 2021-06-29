"""Generate palettes for different tasks."""
from typing import Dict, List

import numpy as np

from .label import drivables, labels, lane_categories
from .to_mask import STUFF_NUM

PALETTES: Dict[str, List[int]] = {}


def get_palette(mode: str) -> List[int]:
    """Generate mapping for the required task."""
    if mode in ["ins_seg", "pan_seg"]:
        palette = (
            np.multiply(np.random.rand(768), 255).astype(np.uint8).tolist()
        )
        palette[:3] = [0, 0, 0]
        if mode == "ins_seg":
            assert isinstance(palette, list)
            return palette

        if mode in PALETTES:
            start, end = 3, (STUFF_NUM + 1) * 3
            palette[start:end] = PALETTES[mode][start:end]
        else:
            for label in labels[1 : STUFF_NUM + 1]:
                id_ = label.id
                palette[id_ * 3 : (id_ + 1) * 3] = label.color[:]
            PALETTES[mode] = palette

    if mode in PALETTES:
        return PALETTES[mode]

    if mode == "seg_track":
        palette = (
            np.multiply(np.random.rand(768), 255).astype(np.uint8).tolist()
        )
        palette[:3] = [0, 0, 0]
    else:
        color_mapping = {
            label.trainId: label.color
            for label in dict(
                sem_seg=labels, drivable=drivables, lane_mark=lane_categories
            )[mode]
        }
        color_mapping[255] = (0, 0, 0)
        palette = [0] * 768
        for id_, color in color_mapping.items():
            palette[id_ * 3 : (id_ + 1) * 3] = color[:]

    PALETTES[mode] = palette
    assert isinstance(palette, list)
    return palette
