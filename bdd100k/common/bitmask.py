"""Support functions for Bitmask."""

from typing import List, Tuple

import numpy as np
from scalabel.common.typing import NDArrayF64, NDArrayI32, NDArrayU8

MAX_DET = 100


def gen_blank_bitmask(shape: Tuple[int, ...]) -> NDArrayU8:
    """Generate blank bitmask given the shape."""
    assert shape[-1] == 4
    bitmask: NDArrayU8 = np.zeros(shape, dtype=np.uint8)
    bitmask[..., 3] = 1  # instance_id as 1
    bitmask[..., 1] = 3  # ignored and crowded
    return bitmask


def parse_bitmask(
    bitmask: NDArrayU8, stacked: bool = False
) -> List[NDArrayI32]:
    """Parse information from bitmasks and compress its value range.

    The compression works like: [4, 2, 9] --> [2, 1, 3]
    """
    bitmask = bitmask.astype(np.int32)
    category_map = bitmask[:, :, 0]
    attributes_map = bitmask[:, :, 1]
    instance_map = (bitmask[:, :, 2] << 8) + bitmask[:, :, 3]

    # 0 is for the background
    instance_ids = np.sort(np.unique(instance_map[instance_map >= 1]))
    category_ids = np.zeros(instance_ids.shape, dtype=instance_ids.dtype)
    attributes = np.zeros(instance_ids.shape, dtype=instance_ids.dtype)

    if not stacked:
        masks: NDArrayI32 = np.zeros(bitmask.shape[:2], dtype=np.int32)
    else:
        masks = np.zeros(
            (*bitmask.shape[:2], len(instance_ids)), dtype=np.int32
        )
    for i, instance_id in enumerate(instance_ids):
        mask_inds_i = instance_map == instance_id
        if not stacked:
            # 0 is for the background
            masks[mask_inds_i] = i + 1
        else:
            masks[mask_inds_i, i] = 1

        attributes_i: NDArrayI32 = np.unique(attributes_map[mask_inds_i])
        assert attributes_i.shape[0] == 1
        attributes[i] = attributes_i[0]

        category_ids_i: NDArrayI32 = np.unique(category_map[mask_inds_i])
        assert category_ids_i.shape[0] == 1
        category_ids[i] = category_ids_i[0]

    return [masks, instance_ids, attributes, category_ids]


def bitmask_intersection_rate(
    gt_masks: NDArrayI32, pred_masks: NDArrayI32
) -> Tuple[NDArrayF64, NDArrayF64]:
    """Returns the intersection over the area of the predicted box."""
    assert gt_masks.shape == pred_masks.shape
    m: int = np.max(gt_masks)
    n: int = min(np.max(pred_masks), MAX_DET)

    gt_masks = gt_masks.reshape(-1)
    pred_masks = pred_masks.reshape(-1)
    pred_masks[pred_masks > MAX_DET] = 0

    confusion: NDArrayI32 = gt_masks * (1 + n) + pred_masks
    bin_num = (1 + m) * (1 + n)
    hist = np.histogram(confusion, bins=bin_num, range=(0, bin_num))[0]
    conf_matrix = hist.reshape(1 + m, 1 + n)
    gt_areas = conf_matrix.sum(axis=1, keepdims=True)[1:, :]
    pred_areas = conf_matrix.sum(axis=0, keepdims=True)[:, 1:]

    inter = conf_matrix[1:, 1:]
    union = gt_areas + pred_areas - inter
    ious = inter / union
    ious = np.where(union > 0, ious, 0.0)
    iofs = inter / pred_areas
    iofs = np.where(pred_areas > 0, iofs, 0.0)
    return ious, iofs
