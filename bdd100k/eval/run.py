"""Evaluation helper functoins."""

import argparse

from scalabel.label.to_coco import group_and_sort

from ..common.utils import (
    DEFAULT_COCO_CONFIG,
    group_and_sort_files,
    list_files,
    read,
)
from .detect import evaluate_det
from .ins_seg import evaluate_ins_seg
from .mot import acc_single_video_mot, evaluate_track
from .mots import acc_single_video_mots
from .seg import evaluate_drivable, evaluate_segmentation


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        choices=["seg", "det", "ins_seg", "drivable", "mot", "mots"],
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
        "--mot-nproc",
        type=int,
        default=4,
        help="number of processes for mot evaluation",
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
    elif args.task == "seg":
        evaluate_segmentation(args.gt, args.result, 19, 17)
    elif args.task == "det":
        evaluate_det(args.gt, args.result, args.config, args.out_dir)
    elif args.task == "ins_seg":
        evaluate_ins_seg(args.gt, args.result, args.score_file, args.out_dir)
    elif args.task == "mot":
        evaluate_track(
            acc_single_video_mot,
            gts=group_and_sort(read(args.gt)),
            results=group_and_sort(read(args.result)),
            iou_thr=args.mot_iou_thr,
            ignore_iof_thr=args.mot_ignore_iof_thr,
            nproc=args.mot_nproc,
        )
    elif args.task == "mots":
        evaluate_track(
            acc_single_video_mots,
            gts=group_and_sort_files(list_files(args.gt)),
            results=group_and_sort_files(list_files(args.result)),
            iou_thr=args.mot_iou_thr,
            ignore_iof_thr=args.mot_ignore_iof_thr,
            nproc=args.mot_nproc,
        )


if __name__ == "__main__":
    run()
