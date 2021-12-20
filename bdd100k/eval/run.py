"""Evaluation helper functions."""

import argparse
import json
import os

from scalabel.common.parallel import NPROC
from scalabel.eval.detect import evaluate_det
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.pose import evaluate_pose
from scalabel.eval.result import Result
from scalabel.eval.sem_seg import evaluate_sem_seg as sc_eval_sem_seg
from scalabel.label.io import group_and_sort, load

from ..common.logger import logger
from ..common.utils import (
    group_and_sort_files,
    list_files,
    load_bdd100k_config,
)
from ..label.to_scalabel import bdd100k_to_scalabel
from .ins_seg import evaluate_ins_seg
from .lane import evaluate_lane_marking
from .mots import evaluate_seg_track
from .pan_seg import evaluate_pan_seg
from .seg import evaluate_drivable, evaluate_sem_seg


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        choices=[
            "det",
            "sem_seg",
            "ins_seg",
            "pan_seg",
            "drivable",
            "lane_mark",
            "box_track",
            "seg_track",
            "pose",
        ],
        required=True,
    )
    parser.add_argument(
        "--gt", "-g", required=True, help="path to ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to results to be evaluated"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="path to the config file",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="iou threshold for mot evaluation",
    )
    parser.add_argument(
        "--ignore-iof-thr",
        type=float,
        default=0.5,
        help="ignore iof threshold for mot evaluation",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for evaluation",
    )
    # Flags for detection and instance segmentation
    parser.add_argument(
        "--out-file", type=str, default=None, help="Path to store output files"
    )
    # Flags for instance segmentation
    parser.add_argument(
        "--score-file",
        type=str,
        default=".",
        help="Path to store the prediction scoring file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="without logging",
    )

    args = parser.parse_args()

    return args


def run() -> None:
    """Main."""
    args = parse_args()
    if args.config is not None:
        bdd100k_config = load_bdd100k_config(args.config)
    elif args.task in ["det", "ins_seg", "box_track", "seg_track", "pose"]:
        bdd100k_config = load_bdd100k_config(args.task)
    else:
        return

    # Determine if the input contains bitmasks or JSON files
    if len(list_files(args.result, ".png")) > 0:
        use_scalabel = False
    elif len(list_files(args.result, ".json")) > 0:
        use_scalabel = True
    else:
        return

    if args.task == "det":
        results: Result = evaluate_det(
            bdd100k_to_scalabel(
                load(args.gt, args.nproc).frames, bdd100k_config
            ),
            bdd100k_to_scalabel(
                load(args.result, args.nproc).frames, bdd100k_config
            ),
            bdd100k_config.scalabel,
            nproc=args.nproc,
        )
    elif args.task == "ins_seg":
        results = evaluate_ins_seg(
            args.gt,
            args.result,
            args.score_file,
            bdd100k_config.scalabel,
            nproc=args.nproc,
        )
    elif args.task == "box_track":
        results = evaluate_track(
            acc_single_video_mot,
            gts=group_and_sort(
                bdd100k_to_scalabel(
                    load(args.gt, args.nproc).frames, bdd100k_config
                )
            ),
            results=group_and_sort(
                bdd100k_to_scalabel(
                    load(args.result, args.nproc).frames, bdd100k_config
                )
            ),
            config=bdd100k_config.scalabel,
            iou_thr=args.iou_thr,
            ignore_iof_thr=args.ignore_iof_thr,
            nproc=args.nproc,
        )
    elif args.task == "seg_track":
        results = evaluate_seg_track(
            gts=group_and_sort_files(
                list_files(args.gt, ".png", with_prefix=True)
            ),
            results=group_and_sort_files(
                list_files(args.result, ".png", with_prefix=True)
            ),
            config=bdd100k_config.scalabel,
            iou_thr=args.iou_thr,
            ignore_iof_thr=args.ignore_iof_thr,
            nproc=args.nproc,
        )
    elif args.task == "pose":
        results = evaluate_pose(
            bdd100k_to_scalabel(
                load(args.gt, args.nproc).frames, bdd100k_config
            ),
            bdd100k_to_scalabel(
                load(args.result, args.nproc).frames, bdd100k_config
            ),
            bdd100k_config.scalabel,
            nproc=args.nproc,
        )

    gt_paths = list_files(args.gt, ".png", with_prefix=True)
    pred_paths = list_files(args.result, ".png", with_prefix=True)
    if args.task == "drivable":
        results = evaluate_drivable(
            gt_paths, pred_paths, nproc=args.nproc, with_logs=not args.quiet
        )
    elif args.task == "lane_mark":
        results = evaluate_lane_marking(
            gt_paths, pred_paths, nproc=args.nproc, with_logs=not args.quiet
        )
    elif args.task == "sem_seg":
        if not use_scalabel:
            results = evaluate_sem_seg(
                gt_paths, pred_paths, nproc=args.nproc, with_logs=not args.quiet
            )
        else:
            results = sc_eval_sem_seg(
                bdd100k_to_scalabel(
                    load(args.gt, args.nproc).frames, bdd100k_config
                ),
                bdd100k_to_scalabel(
                    load(args.result, args.nproc).frames, bdd100k_config
                ),
                bdd100k_config.scalabel,
                nproc=args.nproc,
            )
    elif args.task == "pan_seg":
        results = evaluate_pan_seg(
            gt_paths, pred_paths, nproc=args.nproc, with_logs=not args.quiet
        )
    else:
        return

    logger.info(results)
    if args.out_file:
        out_folder = os.path.split(args.out_file)[0]
        if not os.path.exists(out_folder) and out_folder:
            os.makedirs(out_folder)
        with open(args.out_file, "w", encoding="utf-8") as fp:
            json.dump(dict(results), fp, indent=2)


if __name__ == "__main__":
    run()
