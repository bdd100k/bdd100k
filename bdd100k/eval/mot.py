"""BDD100K tracking evaluation with CLEAR MOT metrics."""
import time
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple, Union, overload

import motmetrics as mm
import numpy as np
import pandas as pd
from scalabel.label.to_coco import box2d_to_bbox
from scalabel.label.typing import Frame, Label

from ..common.logger import logger

EvalResults = Dict[str, Dict[str, float]]
Frames = List[Frame]
Files = List[str]
FramesList = List[Frames]
FilesList = List[Files]
FramesFunc = Callable[[Frames, Frames, float, float], List[mm.MOTAccumulator]]
FilesFunc = Callable[[Files, Files, float, float], List[mm.MOTAccumulator]]


METRIC_MAPS = {
    "idf1": "IDF1",
    "mota": "MOTA",
    "motp": "MOTP",
    "num_false_positives": "FP",
    "num_misses": "FN",
    "num_switches": "IDSw",
    "mostly_tracked": "MT",
    "partially_tracked": "PT",
    "mostly_lost": "ML",
    "num_fragmentations": "FM",
}

SUPER_CLASSES = {
    "HUMAN": ["pedestrian", "rider"],
    "VEHICLE": ["car", "truck", "bus", "train"],
    "BIKE": ["motorcycle", "bicycle"],
}
CLASSES = [c for cs in SUPER_CLASSES.values() for c in cs]
IGNORE_CLASSES = ["trailer", "other person", "other vehicle"]


def parse_objects(objects: List[Label]) -> List[np.ndarray]:
    """Parse objects under Scalable formats."""
    bboxes, labels, ids, ignore_bboxes = [], [], [], []
    for obj in objects:
        box_2d = obj.box_2d
        if box_2d is None:
            continue
        bbox = box2d_to_bbox(box_2d)
        category = obj.category
        if category in CLASSES:
            if obj.attributes is not None and bool(
                obj.attributes.get("crowd", False)
            ):
                ignore_bboxes.append(bbox)
            else:
                bboxes.append(bbox)
                labels.append(CLASSES.index(category))
                ids.append(obj.id)
        elif category in IGNORE_CLASSES:
            ignore_bboxes.append(bbox)
        else:
            raise KeyError("Unknown category.")
    return list(map(np.array, [bboxes, labels, ids, ignore_bboxes]))


def intersection_over_area(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Returns the intersection over the area of the predicted box."""
    out = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            w = min(p[0] + p[2], g[0] + g[2]) - max(p[0], g[0])
            h = min(p[1] + p[3], g[1] + g[3]) - max(p[1], g[1])
            out[i][j] = max(w, 0) * max(h, 0) / float(p[2] * p[3])
    return out


def acc_single_video_mot(
    gts: List[Frame],
    results: List[Frame],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
) -> List[mm.MOTAccumulator]:
    """Accumulate results for one video."""
    num_classes = len(CLASSES)
    assert len(gts) == len(results)
    get_frame_index = (
        lambda x: x.frame_index if x.frame_index is not None else 0
    )
    gts = sorted(gts, key=get_frame_index)
    results = sorted(results, key=get_frame_index)
    accs = [mm.MOTAccumulator(auto_id=True) for i in range(num_classes)]
    for gt, result in zip(gts, results):
        assert gt.frame_index == result.frame_index
        gt_bboxes, gt_labels, gt_ids, gt_ignores = parse_objects(gt.labels)
        pred_bboxes, pred_labels, pred_ids, _ = parse_objects(result.labels)
        for i in range(num_classes):
            gt_inds, pred_inds = gt_labels == i, pred_labels == i
            gt_bboxes_c, gt_ids_c = gt_bboxes[gt_inds], gt_ids[gt_inds]
            pred_bboxes_c, pred_ids_c = (
                pred_bboxes[pred_inds],
                pred_ids[pred_inds],
            )
            if gt_bboxes_c.shape[0] == 0 and pred_bboxes_c.shape[0] != 0:
                distances = np.full((0, pred_bboxes_c.shape[0]), np.nan)
            elif gt_bboxes_c.shape[0] != 0 and pred_bboxes_c.shape[0] == 0:
                distances = np.full((gt_bboxes_c.shape[0], 0), np.nan)
            else:
                distances = mm.distances.iou_matrix(
                    gt_bboxes_c, pred_bboxes_c, max_iou=1 - iou_thr
                )
            if gt_ignores.shape[0] > 0:
                # 1. assign gt and preds
                fps = np.ones(pred_bboxes_c.shape[0]).astype(np.bool8)
                le, ri = mm.lap.linear_sum_assignment(distances)
                for m, n in zip(le, ri):
                    if np.isfinite(distances[m, n]):
                        fps[n] = False
                # 2. ignore by iof
                iofs = intersection_over_area(pred_bboxes_c, gt_ignores)
                ignores = (iofs > ignore_iof_thr).any(axis=1)
                # 3. filter preds
                valid_inds = ~(fps & ignores)
                pred_ids_c = pred_ids_c[valid_inds]
                distances = distances[:, valid_inds]
            if distances.shape != (0, 0):
                accs[i].update(gt_ids_c, pred_ids_c, distances)
    return accs


def aggregate_accs(
    accumulators: List[List[mm.MOTAccumulator]],
) -> Tuple[List[List[str]], List[List[mm.MOTAccumulator]], List[str]]:
    """Aggregate the results of the entire dataset."""
    # accs for each class
    items = CLASSES.copy()
    names: List[List[str]] = [[] for c in CLASSES]
    accs: List[List[str]] = [[] for c in CLASSES]
    for video_ind, _accs in enumerate(accumulators):
        for cls_ind, acc in enumerate(_accs):
            if (
                len(acc._events["Type"])  # pylint: disable=protected-access
                == 0
            ):
                continue
            name = f"{CLASSES[cls_ind]}_{video_ind}"
            names[cls_ind].append(name)
            accs[cls_ind].append(acc)

    # super categories
    for super_cls, classes in SUPER_CLASSES.items():
        items.append(super_cls)
        names.append([n for c in classes for n in names[CLASSES.index(c)]])
        accs.append([a for c in classes for a in accs[CLASSES.index(c)]])

    # overall
    items.append("OVERALL")
    names.append([n for name in names[: len(CLASSES)] for n in name])
    accs.append([a for acc in accs[: len(CLASSES)] for a in acc])

    return names, accs, items


def evaluate_single_class(
    names: List[str], accs: List[mm.MOTAccumulator]
) -> List[float]:
    """Evaluate results for one class."""
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=METRIC_MAPS.keys(), generate_overall=True
    )
    results = [v["OVERALL"] for k, v in summary.to_dict().items()]
    motp_ind = list(METRIC_MAPS).index("motp")
    if np.isnan(results[motp_ind]):
        num_dets = mh.compute_many(
            accs,
            names=names,
            metrics=["num_detections"],
            generate_overall=True,
        )
        sum_motp = (summary["motp"] * num_dets["num_detections"]).sum()
        motp = mm.math_util.quiet_divide(
            sum_motp, num_dets["num_detections"]["OVERALL"]
        )
        results[motp_ind] = float(1 - motp)
    else:
        results[motp_ind] = 1 - results[motp_ind]
    return results


def render_results(
    summaries: List[List[float]],
    items: List[str],
    metrics: List[str],
) -> EvalResults:
    """Render the evaluation results."""
    eval_results = pd.DataFrame(columns=metrics)
    # category, super-category and overall results
    for i, item in enumerate(items):
        eval_results.loc[item] = summaries[i]
    dtypes = {m: type(d) for m, d in zip(metrics, summaries[0])}
    # average results
    avg_results: List[Union[int, float]] = []
    for i, m in enumerate(metrics):
        v = np.array([s[i] for s in summaries[: len(CLASSES)]])
        v = np.nan_to_num(v, nan=0)
        if dtypes[m] == int:
            avg_results.append(int(v.sum()))
        elif dtypes[m] == float:
            avg_results.append(float(v.mean()))
        else:
            raise TypeError()
    eval_results.loc["AVERAGE"] = avg_results
    eval_results = eval_results.astype(dtypes)

    metric_host = mm.metrics.create()
    metric_host.register(mm.metrics.motp, formatter="{:.1%}".format)
    strsummary = mm.io.render_summary(
        eval_results,
        formatters=metric_host.formatters,
        namemap=METRIC_MAPS,
    )
    strsummary = strsummary.split("\n")
    assert len(strsummary) == len(CLASSES) + len(SUPER_CLASSES) + 3
    split_line = "-" * len(strsummary[0])
    strsummary.insert(1, split_line)
    strsummary.insert(2 + len(CLASSES), split_line)
    strsummary.insert(3 + len(CLASSES) + len(SUPER_CLASSES), split_line)
    strsummary = "".join([f"{s}\n" for s in strsummary])
    strsummary = "\n" + strsummary
    logger.info(strsummary)

    outputs: EvalResults = dict()
    for i, item in enumerate(items):
        outputs[item] = dict()
        for j, metric in enumerate(METRIC_MAPS.values()):
            outputs[item][metric] = summaries[i][j]
    outputs["OVERALL"]["mIDF1"] = eval_results.loc["AVERAGE"]["idf1"]
    outputs["OVERALL"]["mMOTA"] = eval_results.loc["AVERAGE"]["mota"]
    outputs["OVERALL"]["mMOTP"] = eval_results.loc["AVERAGE"]["motp"]

    return outputs


@overload
def evaluate_track(
    acc_single_video: FramesFunc,
    gts: FramesList,
    results: FramesList,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    nproc: int = 4,
) -> EvalResults:
    ...


@overload
def evaluate_track(
    acc_single_video: FilesFunc,
    gts: FilesList,
    results: FilesList,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    nproc: int = 4,
) -> EvalResults:
    ...


def evaluate_track(
    acc_single_video: Union[FramesFunc, FilesFunc],
    gts: Union[FramesList, FilesList],
    results: Union[FramesList, FilesList],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    nproc: int = 4,
) -> EvalResults:
    """Evaluate CLEAR MOT metrics for BDD100K."""
    logger.info("BDD100K tracking evaluation with CLEAR MOT metrics.")
    t = time.time()
    assert len(gts) == len(results)
    metrics = list(METRIC_MAPS.keys())

    logger.info("accumulating...")
    pool = Pool(nproc)
    accs = pool.starmap(
        acc_single_video,
        zip(
            gts,
            results,
            [iou_thr for _ in range(len(gts))],
            [ignore_iof_thr for _ in range(len(gts))],
        ),
    )
    names, accs, items = aggregate_accs(accs)

    logger.info("evaluating...")
    summaries = pool.starmap(evaluate_single_class, zip(names, accs))
    pool.close()

    logger.info("rendering...")
    eval_results = render_results(summaries, items, metrics)
    t = time.time() - t
    logger.info("evaluation finishes with %.1f s.", t)
    return eval_results
