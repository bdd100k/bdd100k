"""Evaluation helper functions."""

import argparse
import json
import os
from typing import List, Optional, Tuple

from scalabel.common.parallel import NPROC
from scalabel.eval.boundary import evaluate_boundary
from scalabel.eval.detect import evaluate_det
from scalabel.eval.ins_seg import evaluate_ins_seg as sc_eval_ins_seg
from scalabel.eval.mot import acc_single_video_mot, evaluate_track
from scalabel.eval.mots import acc_single_video_mots
from scalabel.eval.mots import evaluate_seg_track as sc_eval_seg_track
from scalabel.eval.pan_seg import evaluate_pan_seg as sc_eval_pan_seg
from scalabel.eval.pose import evaluate_pose
from scalabel.eval.result import Result
from scalabel.eval.sem_seg import evaluate_sem_seg as sc_eval_sem_seg
from scalabel.label.io import group_and_sort, load
from scalabel.label.typing import Frame

from ..common.logger import logger
from ..common.typing import BDD100KConfig
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
        "--config", "-c", default=None, help="path to the config file"
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
    parser.add_argument(
        "--out-file", type=str, default=None, help="path to store output files"
    )
    parser.add_argument(
        "--score-file",
        type=str,
        default=None,
        help="path to store the prediction scoring file (ins_seg)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="without logging"
    )

    return parser.parse_args()


def run_bitmask(
    config: BDD100KConfig,
    task: str,
    gt_paths: List[str],
    pred_paths: List[str],
    score_file: Optional[str],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    quiet: bool = False,
    nproc: int = NPROC,
) -> Result:
    """Run eval for bitmask input."""
    results: Optional[Result] = None
    if task == "ins_seg":
        assert (
            score_file is not None
        ), "ins_seg evaluation with bitmasks requires score_file"
        results = evaluate_ins_seg(
            gt_paths, pred_paths, score_file, config.scalabel, nproc=nproc
        )
    elif task == "seg_track":
        results = evaluate_seg_track(
            gts=group_and_sort_files(gt_paths),
            results=group_and_sort_files(pred_paths),
            config=config.scalabel,
            iou_thr=iou_thr,
            ignore_iof_thr=ignore_iof_thr,
            nproc=nproc,
        )

    if task == "sem_seg":
        results = evaluate_sem_seg(
            gt_paths, pred_paths, nproc=nproc, with_logs=not quiet
        )
    elif task == "drivable":
        results = evaluate_drivable(
            gt_paths, pred_paths, nproc=nproc, with_logs=not quiet
        )
    elif task == "pan_seg":
        results = evaluate_pan_seg(
            gt_paths, pred_paths, nproc=nproc, with_logs=not quiet
        )
    elif task == "lane_mark":
        results = evaluate_lane_marking(
            gt_paths, pred_paths, nproc=nproc, with_logs=not quiet
        )

    assert (
        results is not None
    ), f"{task} evaluation with bitmask format not supported!"

    return results


def run_rle(
    config: BDD100KConfig,
    task: str,
    gt_frames: List[Frame],
    pred_frames: List[Frame],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    nproc: int = NPROC,
) -> Result:
    """Run eval for RLE input."""
    results: Optional[Result] = None
    if task == "ins_seg":
        results = sc_eval_ins_seg(
            gt_frames, pred_frames, config.scalabel, nproc=nproc
        )
    elif task == "seg_track":
        results = sc_eval_seg_track(
            acc_single_video_mots,
            gts=group_and_sort(gt_frames),
            results=group_and_sort(pred_frames),
            config=config.scalabel,
            iou_thr=iou_thr,
            ignore_iof_thr=ignore_iof_thr,
            nproc=nproc,
        )
    elif task in ("sem_seg", "drivable"):
        results = sc_eval_sem_seg(
            gt_frames, pred_frames, config.scalabel, nproc=nproc
        )
    elif task == "pan_seg":
        results = sc_eval_pan_seg(
            gt_frames, pred_frames, config.scalabel, nproc=nproc
        )
    elif task == "lane_mark":
        results = evaluate_boundary(
            gt_frames, pred_frames, config.scalabel, nproc=nproc
        )

    assert (
        results is not None
    ), f"{task} evaluation with RLE format not supported!"

    return results


def _load_frames(
    gt_base: str, result_path: str, config: BDD100KConfig, nproc: int = NPROC
) -> Tuple[List[Frame], List[Frame]]:
    """Load ground truth and prediction frames."""
    gt_frames = bdd100k_to_scalabel(load(gt_base, nproc).frames, config)
    result_frames = bdd100k_to_scalabel(
        load(result_path, nproc).frames, config
    )
    return gt_frames, result_frames


def run() -> None:
    """Main."""
    args = parse_args()
    if args.config is not None:
        bdd100k_config = load_bdd100k_config(args.config)
    else:
        bdd100k_config = load_bdd100k_config(args.task)

    if args.task in ["det", "box_track", "pose"]:
        gt_frames, result_frames = _load_frames(
            args.gt, args.result, bdd100k_config, args.nproc
        )
        if args.task == "det":
            results: Result = evaluate_det(
                gt_frames,
                result_frames,
                bdd100k_config.scalabel,
                nproc=args.nproc,
            )
        elif args.task == "box_track":
            results = evaluate_track(
                acc_single_video_mot,
                gts=group_and_sort(gt_frames),
                results=group_and_sort(result_frames),
                config=bdd100k_config.scalabel,
                iou_thr=args.iou_thr,
                ignore_iof_thr=args.ignore_iof_thr,
                nproc=args.nproc,
            )
        else:  # pose
            results = evaluate_pose(
                gt_frames,
                result_frames,
                bdd100k_config.scalabel,
                nproc=args.nproc,
            )
    else:
        assert os.path.exists(args.gt) and os.path.exists(args.result)
        # for segmentation tasks, determine if the input contains bitmasks or
        # JSON files and call corresponding evaluation function
        res_files = list_files(args.result)
        if len(res_files) > 0 and all(f.endswith(".png") for f in res_files):
            gt_paths = list_files(args.gt, ".png", with_prefix=True)
            pred_paths = list_files(args.result, ".png", with_prefix=True)
            results = run_bitmask(
                bdd100k_config,
                args.task,
                gt_paths,
                pred_paths,
                args.score_file,
                args.iou_thr,
                args.ignore_iof_thr,
                args.quiet,
                args.nproc,
            )
        elif args.result.endswith(".json") or all(
            f.endswith(".json") for f in res_files
        ):
            gt_frames, result_frames = _load_frames(
                args.gt, args.result, bdd100k_config, args.nproc
            )
            results = run_rle(
                bdd100k_config,
                args.task,
                gt_frames,
                result_frames,
                args.iou_thr,
                args.ignore_iof_thr,
                args.nproc,
            )
        else:
            raise ValueError(
                "Input should either be a directory of only bitmasks or a "
                "JSON file / directory of only JSON files"
            )

    logger.info(results)
    if args.out_file:
        out_folder = os.path.split(args.out_file)[0]
        if not os.path.exists(out_folder) and out_folder:
            os.makedirs(out_folder)
        with open(args.out_file, "w", encoding="utf-8") as fp:
            json.dump(dict(results), fp, indent=2)


if __name__ == "__main__":
    run()
