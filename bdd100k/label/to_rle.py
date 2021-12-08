"""Convert bitmask to RLE."""
import argparse
import os
from typing import Callable, Dict, List

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.label.io import load, save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import Category, Dataset, Frame, Label
from scalabel.label.utils import get_leaf_categories

from ..common.typing import BDD100KConfig
from ..common.utils import list_files, load_bdd100k_config
from ..eval.ins_seg import parse_res_bitmask

ToRLEFunc = Callable[[List[Frame], str, List[Category]], None]


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--input",
        help=(
            "directory of bitmasks to convert"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save scalabel formatted label file with RLEs",
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
        help="conversion mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="configuration for the chosen mode",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def insseg_to_rle(frames: List[Frame], input: str, categories: List[Category]) -> None:
    ann_score = {}

    for frame in frames:
        img_name = frame.name.replace(".jpg", ".png")
        ann_score[img_name] = []
        bitmask = np.array(
            Image.open(os.path.join(input, img_name)),
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


def semseg_to_rle(frames: List[Frame], input: str, categories: List[Category]) -> None:
    instance_id = 1
    for frame in frames:
        frame.labels = []
        img_name = frame.name
        bitmask = np.array(
            Image.open(os.path.join(input, img_name)),
            dtype=np.uint8,
        )
        category_ids = np.unique(bitmask[bitmask > 0])

        for category_id in category_ids:
            label = Label(id=str(instance_id))
            label.category = categories[category_id - 1].name
            label.rle = mask_to_rle((bitmask == category_id).astype(np.uint8))
            frame.labels.append(label)
            instance_id += 1


def main() -> None:
    args = parse_args()

    assert os.path.isdir(args.input)

    dataset = load(args.input, args.nproc)
    if args.config is not None:
        bdd100k_config = load_bdd100k_config(args.config)
    elif dataset.config is not None:
        bdd100k_config = BDD100KConfig(config=dataset.config)
    else:
        bdd100k_config = load_bdd100k_config(args.mode)

    categories = get_leaf_categories(bdd100k_config.scalabel.categories)

    convert_funcs: Dict[str, ToRLEFunc] = dict(
        sem_seg=semseg_to_rle,
        ins_seg=insseg_to_rle,
    )

    if args.mode == "ins_seg":
        assert args.score_file is not None
        frames = load(args.score_file).frames

        assert all(
            os.path.exists(os.path.join(args.input, frame.name)) for frame in frames
        ), "Missing some bitmasks."
    else:  # sem_seg
        files = list_files(args.input)
        frames = []
        for file in files:
            if not file.endswith(".png"):
                continue
            frame = Frame(name=file.split('/')[1], labels=[])
            frames.append(frame)

    convert_funcs[args.mode](dataset.frames, args.input, categories)
    save(args.output, frames)


if __name__ == "__main__":
    main()
