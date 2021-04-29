"""Generate palettes for different tasks."""
from typing import Dict, List

import numpy as np

from .label import drivables, labels

PALETTES: Dict[str, List[int]] = {}


def get_palette(mode: str) -> List[int]:
    """Generate mapping for the required task."""
    if mode == "ins_seg":
        palette = (np.random.rand(768) * 255).astype(np.uint8).tolist()
        palette[:3] = [0, 0, 0]
        assert isinstance(palette, list)
        return palette

    if mode in PALETTES:
        return PALETTES[mode]

    if mode == "seg_track":
        palette = (np.random.rand(768) * 255).astype(np.uint8).tolist()
        palette[:3] = [0, 0, 0]
    else:
        color_mapping = {
            label.trainId: label.color
            for label in dict(sem_seg=labels, drivable=drivables)[mode]
        }
        color_mapping[255] = (0, 0, 0)
        palette = [0] * 768
        for id_, color in color_mapping.items():
            palette[id_ * 3 : (id_ + 1) * 3] = color[:]

    PALETTES[mode] = palette
    assert isinstance(palette, list)
    return palette
