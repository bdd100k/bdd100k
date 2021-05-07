"""Evaluation helper functoins."""

import argparse

from scalabel.label.io import group_and_sort, load

from ..common.utils import (
    DEFAULT_COCO_CONFIG,
    group_and_sort_files,
    list_files,
)
from .detect import evaluate_det
from .ins_seg import evaluate_ins_seg
from .lane import evaluate_lane_marking
from .mot import acc_single_video_mot, evaluate_track
from .mots import acc_single_video_mots
from .seg import evaluate_drivable, evaluate_segmentation


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
            "drivable",
            "lane_mark",
            "box_track",
            "seg_track",
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
        default=DEFAULT_COCO_CONFIG,
        help="path to the config file",
    )
    parser.add_argument(
        "--mot-iou-thr",
        type=float,
        default=0.5,
        help="iou threshold for mot evaluation",
    )
    parser.add_argument(
        "--mot-ignore-iof-thr",
        type=float,
        default=0.5,
        help="ignore iof threshold for mot evaluation",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for evaluation",
    )
    # Flags for detection and instance segmentation
    parser.add_argument(
        "--out-dir", type=str, default=".", help="Path to store output files"
    )
    # Flags for instance segmentatoin
    parser.add_argument(
        "--score-file",
        type=str,
        default=".",
        help="Path to store the prediction scoring file",
    )

    args = parser.parse_args()

    return args


def run() -> None:
    """Main."""
    args = parse_args()

    if args.task == "drivable":
        evaluate_drivable(args.gt, args.result)
    elif args.task == "lane_mark":
        evaluate_lane_marking(args.gt, args.result, [1, 2, 5, 10], args.nproc)
    elif args.task == "sem_seg":
        evaluate_segmentation(args.gt, args.result)
    elif args.task == "det":
        evaluate_det(
            args.gt, args.result, args.config, args.out_dir, args.nproc
        )
    elif args.task == "ins_seg":
        evaluate_ins_seg(
            args.gt,
            args.result,
            args.score_file,
            args.config,
            args.out_dir,
            args.nproc,
        )
    elif args.task == "box_track":
        evaluate_track(
            acc_single_video_mot,
            gts=group_and_sort(load(args.gt, args.nproc)),
            results=group_and_sort(load(args.result, args.nproc)),
            iou_thr=args.mot_iou_thr,
            ignore_iof_thr=args.mot_ignore_iof_thr,
            nproc=args.nproc,
        )
    elif args.task == "seg_track":
        evaluate_track(
            acc_single_video_mots,
            gts=group_and_sort_files(
                list_files(args.gt, ".png", with_prefix=True)
            ),
            results=group_and_sort_files(
                list_files(args.result, ".png", with_prefix=True)
            ),
            iou_thr=args.mot_iou_thr,
            ignore_iof_thr=args.mot_ignore_iof_thr,
            nproc=args.nproc,
        )


if __name__ == "__main__":
    run()
