"""Convert mask/bitmask to RLE."""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayU8
from scalabel.label.io import load, save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import Category, Frame, Label
from scalabel.label.utils import get_leaf_categories
from tqdm import tqdm

from ..common.bitmask import parse_bitmask
from ..common.typing import BDD100KConfig
from ..common.utils import list_files, load_bdd100k_config
from ..eval.ins_seg import parse_res_bitmask

ToRLEFunc = Callable[[Frame, str, List[Category]], Frame]


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="mask to RLE conversion")
    parser.add_argument(
        "-i", "--input", help=("directory of bitmasks to convert")
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save scalabel formatted label file with RLEs",
    )
    parser.add_argument("-s", "--score-file", help="path to score file")
    parser.add_argument(
        "-m",
        "--mode",
        default="ins_seg",
        choices=["ins_seg", "sem_seg", "drivable", "seg_track"],
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
    frame: Frame, input_dir: str, categories: List[Category]
) -> Frame:
    """Convert ins_seg bitmasks to rle."""
    ann_score: Dict[str, List[Tuple[int, float]]] = {}
    img_name = frame.name.replace(".jpg", ".png")
    ann_score[img_name] = []
    bitmask: NDArrayU8 = np.array(
        Image.open(os.path.join(input_dir, img_name)),
        dtype=np.uint8,
    )

    if frame.labels is None:
        return frame
    for label in frame.labels:
        assert label.index is not None
        assert label.score is not None
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
        label.rle = mask_to_rle((masks == ann_id).astype(np.uint8))
        labels.append(label)
    frame.labels = labels
    return frame


def semseg_to_rle(
    frame: Frame, input_dir: str, categories: List[Category]
) -> Frame:
    """Convert sem_seg bitmasks to rle."""
    frame.labels = []
    img_name = frame.name.replace(".jpg", ".png")
    bitmask: NDArrayU8 = np.array(
        Image.open(os.path.join(input_dir, img_name)),
        dtype=np.uint8,
    )
    category_ids: NDArrayU8 = np.unique(bitmask)

    label_id = 0
    for category_id in category_ids:
        if category_id >= len(categories):
            continue
        label = Label(id=str(label_id))
        label.category = categories[category_id].name
        label.rle = mask_to_rle((bitmask == category_id).astype(np.uint8))
        frame.labels.append(label)
        label_id += 1

    return frame


def segtrack_to_rle(
    frame: Frame, input_dir: str, categories: List[Category]
) -> Frame:
    """Convert seg_track bitmasks to rle."""
    frame.labels = []
    img_name = frame.name.replace(".jpg", ".png")
    bitmask: NDArrayU8 = np.array(
        Image.open(os.path.join(input_dir, img_name)),
        dtype=np.uint8,
    )
    masks, instance_ids, _, category_ids = parse_bitmask(bitmask)

    # video parameters
    frame.name = frame.name.split("/")[-1]
    frame.videoName = img_name.split("/")[0]
    frame.frameIndex = int(img_name.split("-")[-1].split(".")[0]) - 1

    for i, _ in enumerate(instance_ids):
        label = Label(id=str(instance_ids[i]))
        label.category = categories[category_ids[i] - 1].name
        label.rle = mask_to_rle((masks == i + 1).astype(np.uint8))
        frame.labels.append(label)

    return frame


def main() -> None:
    """Main."""
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
        ins_seg=insseg_to_rle,
        sem_seg=semseg_to_rle,
        drivable=semseg_to_rle,
        seg_track=segtrack_to_rle,
    )

    if args.mode == "ins_seg":
        assert args.score_file is not None
        frames = load(args.score_file).frames

        assert all(
            os.path.exists(
                os.path.join(args.input, frame.name.replace(".jpg", ".png"))
            )
            for frame in frames
        ), "Missing some bitmasks."
    elif args.mode in ("sem_seg", "drivable", "seg_track"):
        files = list_files(args.input)
        frames = []
        for file in files:
            if not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            frame = Frame(name=file.replace(".png", ".jpg"), labels=[])
            frames.append(frame)
    else:
        return

    if args.nproc > 1:
        with Pool(args.nproc) as pool:
            frames = pool.map(
                partial(
                    convert_funcs[args.mode],
                    input_dir=args.input,
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
