"""Convert bitmask to RLE."""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.label.io import load, save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import Category, Frame, Label
from scalabel.label.utils import get_leaf_categories
from tqdm import tqdm

from ..common.typing import BDD100KConfig
from ..common.utils import list_files, load_bdd100k_config
from ..eval.ins_seg import parse_res_bitmask

ToRLEFunc = Callable[[Frame, str, List[Category]], None]

INSTANCE_ID = 1


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--input",
        help=("directory of bitmasks to convert"),
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


def insseg_to_rle(
    frame: Frame, input: str = "", categories: List[Category] = []
) -> Frame:
    ann_score = {}
    img_name = frame.name.replace(".jpg", ".png")
    ann_score[img_name] = []
    bitmask = np.array(
        Image.open(os.path.join(input, img_name)),
        dtype=np.uint8,
    )

    if frame.labels is None:
        return
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
    return frame


def semseg_to_rle(
    frame: Frame, input: str = "", categories: List[Category] = []
) -> Frame:
    frame.labels = []
    img_name = frame.name
    bitmask = np.array(
        Image.open(os.path.join(input, img_name)),
        dtype=np.uint8,
    )
    category_ids = np.unique(bitmask[bitmask > 0])

    global INSTANCE_ID
    for category_id in category_ids:
        label = Label(id=str(INSTANCE_ID))
        label.category = categories[category_id - 1].name
        label.rle = mask_to_rle((bitmask == category_id).astype(np.uint8))
        frame.labels.append(label)
        INSTANCE_ID += 1

    return frame


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
            os.path.exists(os.path.join(args.input, frame.name.replace(".jpg", ".png")))
            for frame in frames
        ), "Missing some bitmasks."
    else:  # sem_seg
        files = list_files(args.input)
        frames = []
        for file in files:
            if not file.endswith(".png"):
                continue
            frame = Frame(name=file, labels=[])
            frames.append(frame)

    if args.nproc > 1:
        with Pool(args.nproc) as pool:
            frames = pool.map(
                partial(
                    convert_funcs[args.mode],
                    input=args.input,
                    categories=categories,
                ),
                tqdm(frames),
            )
    else:
        frames = [
            convert_funcs[args.mode](frame, args.input, categories)
            for frame in tqdm(frames)
        ]

    save(args.output, frames)


if __name__ == "__main__":
    main()
