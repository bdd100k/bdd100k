"""Evaluation helper functoins."""

import argparse
import glob
import json
import os
import os.path as osp
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..common.logger import logger
from ..common.typing import DictAny
from .detect import evaluate_det
from .mot import evaluate_mot


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        choices=["seg", "det", "drivable", "mot"],
        required=True,
    )
    parser.add_argument(
        "--gt", "-g", required=True, help="path to ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to results to be evaluated"
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
    # Flags for detection
    parser.add_argument(
        "--out-dir", type=str, default=".", help="Path to store output files"
    )
    parser.add_argument(
        "--ann-format",
        type=str,
        choices=["coco", "scalabel"],
        default="scalabel",
        help="ground truth annotation format",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["det", "track"],
        default="det",
        help="choose the detection set or the tracking set",
    )

    args = parser.parse_args()

    return args


def fast_hist(
    groundtruth: np.ndarray, prediction: np.ndarray, size: int
) -> np.ndarray:
    """Compute the histogram."""
    k = (groundtruth >= 0) & (groundtruth < size)
    return np.bincount(
        size * groundtruth[k].astype(int) + prediction[k], minlength=size ** 2
    ).reshape(size, size)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    """Calculate per class iou."""
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_png(folder: str) -> List[str]:
    """List png files."""
    paths = []
    for root, _, files in os.walk(folder, topdown=True):
        paths.extend(
            [osp.join(root, f) for f in files if osp.splitext(f)[1] == ".png"]
        )
    return paths


def evaluate_segmentation(
    gt_dir: str, result_dir: str, num_classes: int, key_length: int
) -> None:
    """Evaluate segmentation IoU from input folders."""
    gt_dict = {osp.split(p)[1][:key_length]: p for p in find_all_png(gt_dir)}
    result_dict = {
        osp.split(p)[1][:key_length]: p for p in find_all_png(result_dir)
    }
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError(
            "Result folder only has {} of {} ground truth files.".format(
                len(result_gt_keys), len(gt_dict)
            )
        )
    logger.info("Found %d results", len(result_dict))
    logger.info("Evaluating %d results", len(gt_dict))
    hist = np.zeros((num_classes, num_classes))
    i = 0
    gt_id_set = set()
    for key in sorted(gt_dict.keys()):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, "r"))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, "r"))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        i += 1
        if i % 100 == 0:
            logger.info("Finished %d %f", i, per_class_iu(hist) * 100)
    gt_id_set.remove([255])
    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])

    logger.info(
        "{:.2f}".format(miou),
        ", ".join(["{:.2f}".format(n) for n in list(ious)]),
    )


def evaluate_drivable(gt_dir: str, result_dir: str) -> None:
    """Evaluate drivable area."""
    evaluate_segmentation(gt_dir, result_dir, 3, 17)


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate AP."""
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap  # type: ignore[no-any-return]


def group_by_key(detections: List[DictAny], key: str) -> DictAny:
    """Group detection results by input key."""
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


def cat_pc(
    groundtruth: List[DictAny],
    predictions: List[DictAny],
    thresholds: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refers to https://github.com/rbgirshick/py-faster-rcnn."""
    num_gts = len(groundtruth)
    image_gts = group_by_key(groundtruth, "name")
    image_gt_boxes = {
        k: np.array([[float(z) for z in b["bbox"]] for b in boxes])
        for k, boxes in image_gts.items()
    }
    image_gt_checked = {
        k: np.zeros((len(boxes), len(thresholds)))
        for k, boxes in image_gts.items()
    }
    predictions = sorted(
        predictions, key=lambda x: float(x["score"]), reverse=True
    )

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p["bbox"]
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p["name"]]
            gt_checked = image_gt_checked[p["name"]]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                + (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0)
                * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.0
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.0
            else:
                fp[i, t] = 1.0

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])

    return recalls, precisions, ap


def evaluate_detection(gt_path: str, result_path: str) -> None:
    """Evaluate detection results."""
    gt = json.load(open(gt_path, "r"))
    pred = json.load(open(result_path, "r"))
    cat_gt = group_by_key(gt, "category")
    cat_pred = group_by_key(pred, "category")
    cat_list = sorted(cat_gt.keys())
    thresholds = [0.75]
    aps = np.zeros((len(thresholds), len(cat_list)))
    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)[2]
            aps[:, i] = ap
    aps *= 100
    m_ap = np.mean(aps)
    mean, breakdown = m_ap, aps.flatten().tolist()

    logger.info(
        "{:.2f}".format(mean),
        ", ".join(["{:.2f}".format(n) for n in breakdown]),
    )


def read(inputs: str) -> List[List[DictAny]]:
    """Read annotations from file/files."""
    if osp.isdir(inputs):
        files = glob.glob(osp.join(inputs, "*.json"))
        outputs = [json.load(open(file)) for file in files]
    elif osp.isfile(inputs) and inputs.endswith("json"):
        outputs = json.load(open(inputs))
    else:
        raise TypeError("Inputs must be a folder or a JSON file.")
    outputs = sorted(outputs, key=lambda x: str(x[0]["video_name"]))
    return outputs


def run() -> None:
    """Main."""
    args = parse_args()

    if args.task == "drivable":
        evaluate_drivable(args.gt, args.result)
    elif args.task == "seg":
        evaluate_segmentation(args.gt, args.result, 19, 17)
    elif args.task == "det":
        evaluate_det(
            args.gt, args.result, args.out_dir, args.ann_format, args.mode
        )
    elif args.task == "mot":
        evaluate_mot(
            gts=read(args.gt),
            results=read(args.result),
            iou_thr=args.mot_iou_thr,
            ignore_iof_thr=args.mot_ignore_iof_thr,
            nproc=args.mot_nproc,
        )


if __name__ == "__main__":
    run()
