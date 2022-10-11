"""Scripts for running COLMAP on BDD100k or any sequences."""
import argparse
import json
import os
import pdb
from typing import List, Optional
import numpy as np
from .run_postprocess import postprocess


def initialize_postprocess_progress_tracker(target_path, save_path):
    """Create progress tracker for depth postprocess."""
    singles_list = os.listdir(os.path.join(target_path, "singles"))
    overlaps_list = [
        int(i) for i in os.listdir(os.path.join(target_path, "overlaps"))
    ]
    overlaps_list.sort()
    singles_list.sort()
    overlaps_dense_list = []
    singles_dense_list = []

    for i in singles_list:
        single_denses_path = os.path.join(
            target_path, "singles", str(i), "dense"
        )
        for j in os.listdir(single_denses_path):
            path = os.path.join(single_denses_path, j)
            singles_dense_list.append(path)

    for i in overlaps_list:
        overlap_denses_path = os.path.join(
            target_path, "overlaps", str(i), "dense"
        )
        for j in os.listdir(overlap_denses_path):
            path = os.path.join(overlap_denses_path, j)
            overlaps_dense_list.append(path)

    progress_tracker = {
        "Finished?": False,
        "Done": [],
        "Doing": [],
        "Failed": [],
        "Left": singles_dense_list + overlaps_dense_list,
        "All": singles_dense_list + overlaps_dense_list,
    }
    with open(save_path, "w", encoding="utf8") as fp:
        json.dump(progress_tracker, fp, indent=2)


def is_progress_finished(progress_path):
    """Check if we are done in the progress tracker."""
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)

    return progress["Finished?"] or len(progress["Left"]) == 0


def failure_update(progress_path, element):
    """Update when postprocessing failed."""
    with open(progress_path) as fp:
        progress = json.load(fp)
    try:
        progress["Doing"].remove(element)
    except:
        print(f"This element {element} is not in Doing")
        return
    progress["Failed"].append(element)
    if len(progress["Left"]) == 0 and len(progress["Doing"]) == 0:
        progress["Finished?"] = True
    with open(progress_path, "w") as fp:
        json.dump(progress, fp, indent=2)


def progress_update(progress_path, element):
    """Update successful postprocessing"""
    with open(progress_path) as fp:
        progress = json.load(fp)
    try:
        progress["Doing"].remove(element)
    except:
        print(f"This element {element} is not in Doing")
        return
    progress["Done"].append(element)
    if len(progress["Left"]) == 0 and len(progress["Doing"]) == 0:
        progress["Finished?"] = True
    with open(progress_path, "w") as fp:
        json.dump(progress, fp, indent=2)


def progress_get_next(progress_path, check_path=None):
    """Get the next target to proceed in progress tracker."""
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)
    next_element = progress["Left"].pop(0)
    progress["Doing"].append(next_element)
    with open(progress_path, "w", encoding="utf8") as fp:
        json.dump(progress, fp, indent=2)
    return next_element


def parse_args():
    """All Arguments"""
    parser = argparse.ArgumentParser(
        description="Postprocess depth map for BDD100k data"
    )
    parser.add_argument(
        "--postcode",
        type=str,
        help="Which postcode to run. e.g. 10009",
    )
    parser.add_argument(
        "--timeofday",
        type=str,
        default="daytime",
        help="Which timeofday to process(daytime, night, dawn_dusk)",
    )
    parser.add_argument(
        "--target-path",
        "-t",
        type=str,
        default="/srv/beegfs02/scratch/bdd100k/data/sfm/postcode",
        help="Local path to save all postcodes and the results.",
    )
    parser.add_argument(
        "--min_depth_percentile",
        help="minimum visualization depth percentile",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="maximum visualization depth percentile",
        type=float,
        default=95,
    )
    parser.add_argument(
        "--min_depth",
        help="minimum visualization depth in meter",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--max_depth",
        help="maximum visualization depth in meter",
        type=float,
        default=80,
    )
    args = parser.parse_args()
    return args


def main():
    """Postprocess depth"""
    args = parse_args()
    postcode_path = os.path.join(args.target_path, args.postcode)
    timeofday_path = os.path.join(postcode_path, args.timeofday)
    postprocess_progress = os.path.join(
        timeofday_path, "postprocess_progress.json"
    )

    if not os.path.exists(postprocess_progress):
        initialize_postprocess_progress_tracker(
            timeofday_path, postprocess_progress
        )

    while not is_progress_finished(postprocess_progress):
        dense_path = progress_get_next(postprocess_progress)
        target_path = os.path.dirname(os.path.dirname(dense_path))
        print("Start postprocessing for: " + dense_path)
        postprocess(dense_path, target_path, args)
        progress_update(postprocess_progress, dense_path)


if __name__ == "__main__":
    main()
