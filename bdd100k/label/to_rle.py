"""Convert bitmask to RLE."""
import argparse
import json
import os
from typing import List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayU8
from scalabel.label.io import load, save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import RLE, Dataset, Frame, Label
from scalabel.label.utils import get_leaf_categories

from ..common.bitmask import parse_bitmask
from ..common.typing import BDD100KConfig
from ..common.utils import list_files, load_bdd100k_config
from ..eval.ins_seg import parse_res_bitmask
from .to_scalabel import bdd100k_to_scalabel


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
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
        "-s",
        "--score-file",
        help="path to score file",
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
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert args.config is not None
    assert args.input is not None
    assert args.output is not None

    dataset = load(args.input, args.nproc)
    if args.config is not None:
        bdd100k_config = load_bdd100k_config(args.config)
    elif dataset.config is not None:
        bdd100k_config = BDD100KConfig(config=dataset.config)
    else:
        bdd100k_config = load_bdd100k_config(args.mode)

    categories = get_leaf_categories(bdd100k_config.scalabel.categories)

    if args.mode == "ins_seg":
        assert args.score_file is not None

        dataset = load(args.score_file)
        ann_score = {}

        for frame in dataset.frames:
            img_name = frame.name.replace(".jpg", ".png")
            ann_score[img_name] = []
            bitmask = np.array(
                Image.open(os.path.join(args.input, img_name)),
                dtype=np.uint8,
            )

            if frame.labels is None:
                continue
            for label in frame.labels:
                ann_score[img_name].append((label.index, label.score))

            masks, ann_ids, scores, category_ids = parse_res_bitmask(
                ann_score[img_name], bitmask
            )

            labels = []
            for ann_id in ann_ids:
                label = Label(
                    id=ann_id,
                    category=categories[category_ids[ann_id - 1] - 1].name,
                    score=scores[ann_id - 1],
                )

                # RLE
                label.rle = mask_to_rle((masks == ann_id).astype(np.uint8))

                labels.append(label)
            frame.labels = labels
    else:  # sem_seg
        files = list_files(args.input)
        frames = []
        for file in files:
            if not file.endswith(".png"):
                continue
            frame = Frame(name=file.split('/')[1], labels=[])
            frames.append(frame)

        dataset = Dataset(frames=frames)

        instance_id = 1
        for frame in dataset.frames:
            img_name = frame.name
            bitmask = np.array(
                Image.open(os.path.join(args.input, img_name)),
                dtype=np.uint8,
            )
            category_ids = np.unique(bitmask[bitmask > 0])

            for category_id in category_ids:
                label = Label(id=str(instance_id))
                label.category = categories[category_id - 1]
                label.rle = mask_to_rle((bitmask == category_id).astype(np.uint8))
                frame.labels.append(label)
                instance_id += 1

    save(args.output, dataset.frames)


if __name__ == "__main__":
    main()
