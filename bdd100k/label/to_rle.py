"""Convert bitmask to RLE."""
import argparse
import json
import os
from typing import List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scalabel.common.typing import NDArrayU8
from scalabel.label.io import load, save
from scalabel.label.typing import RLE, Dataset, Frame, Label

from ..common.bitmask import parse_bitmask
from ..common.utils import list_files, load_bdd100k_config
from ..eval.ins_seg import parse_res_bitmask


def get_categories(config: str) -> List[str]:
    """Load category configs by mode."""
    # TODO: Category for sem_seg is stale
    categories = []
    for c in load_bdd100k_config(config).scalabel.categories:
        if c.subcategories:
            categories.extend([s.name for s in c.subcategories])
        else:
            categories.append(c.name)
    return categories


def mask_to_rle(mask: NDArrayU8) -> RLE:
    assert np.count_nonzero(mask) > 0
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle_label = dict(counts=rle["counts"].decode("utf-8"), size=rle["size"])
    return RLE(**rle_label)


def img_to_label_id(img_name: str, idx: int) -> str:
    id = img_name.split('.')[0]
    if idx < 10:
        return f"{id}-0{idx}"
    return f"{id}-{idx}"


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--input",
        help=(
            "root directory of bdd100k label Json files or path to a label " "json file"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco formatted label file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="ins_seg",
        choices=[
            "sem_seg",
            "ins_seg",
        ],
        help="conversion mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration for COCO categories",
    )
    args = parser.parse_args()

    assert args.config is not None
    assert args.input is not None
    assert args.output is not None

    categories = get_categories(args.config)

    if args.mode == "ins_seg":
        dataset = load(args.input)
        ann_score = {}

        for frame in dataset.frames:
            img_name = frame.name.replace(".jpg", ".png")
            ann_score[img_name] = []
            bitmask = np.array(
                Image.open(os.path.join(args.input, "bitmasks", img_name)),
                dtype=np.uint8,
            )

            if frame.labels is None:
                continue
            for label in frame.labels:
                ann_score[img_name].append((label.index, label.score))

            masks, ann_ids, _, category_ids = parse_res_bitmask(
                ann_score[img_name], bitmask
            )

            for idx, id in enumerate(ann_ids):
                label = frame.labels[id - 1]
                label.category = categories[category_ids[idx] - 1]

                # RLE
                label.rle = mask_to_rle((masks == idx + 1).astype(np.uint8))
    else:  # sem_seg
        files = list_files(args.input)
        frames = []
        for file in files:
            if not file.endswith(".png"):
                continue
            frame = Frame(name=file.split('/')[1], labels=[])
            frames.append(frame)

        dataset = Dataset(frames=frames)

        for frame in dataset.frames:
            img_name = frame.name
            bitmask = np.array(
                Image.open(os.path.join(args.input, "masks", img_name)),
                dtype=np.uint8,
            )
            category_ids = np.unique(bitmask[bitmask > 0])

            for idx, id in enumerate(category_ids):
                label = Label(id=img_to_label_id(img_name, idx))
                label.category = categories[id - 1]
                label.rle = mask_to_rle((bitmask == id).astype(np.uint8))
                frame.labels.append(label)

    save(args.output, dataset.frames)


if __name__ == "__main__":
    main()
