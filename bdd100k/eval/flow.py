"""Evaluation code for optical flow."""
import argparse
import copy
import json
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scalabel.common.io import open_write_text
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64, NDArrayI32, NDArrayU8
from scalabel.eval.result import AVERAGE, Result, Scores
from scalabel.eval.sem_seg import (
    fast_hist,
    per_class_acc,
    per_class_iou,
    safe_divide,
)
from scalabel.label.io import group_and_sort, load
from scalabel.label.transforms import rle_to_mask
from scalabel.label.typing import Config, Frame, ImageSize, Label
from tqdm import tqdm

from ..common.logger import logger
from ..common.utils import load_bdd100k_config
from ..label.to_mask import IGNORE_LABEL

IGNORE_W, IGNORE_H = 64, 36


class FlowResult(Result):
    """The class for optical flow evaluation results."""

    Accuracy: List[Dict[str, float]]
    mAccuracy: List[Dict[str, float]]
    mIoU: List[Dict[str, float]]
    mPrecision: List[Dict[str, float]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "FlowResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert data into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            summary_dict[metric] = scores_list[-1][AVERAGE]
        return summary_dict


def read_flow(name: str) -> NDArrayF64:
    """Read flow file with the suffix '.flo'."""
    with open(name, "rb") as f:
        header = f.read(4)
        if header.decode("utf-8") != "PIEH":
            raise Exception("Flow file header does not contain PIEH")
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape(
            (height, width, 2)
        )
    return flow


def mesh_grid(batchsize: int, height: int, width: int) -> torch.Tensor:
    """Create mesh grid given tensor size."""
    b, h, w = batchsize, height, width
    x_base = torch.arange(0, w).repeat(b, h, 1)  # BHW
    y_base = torch.arange(0, h).repeat(b, w, 1).transpose(1, 2)  # BHW
    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid: torch.Tensor) -> torch.Tensor:
    """Normalize grid to [-1, 1] range."""
    _, _, h, w = v_grid.size()
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (w - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (h - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(
    feat: torch.Tensor,
    flow: torch.Tensor,
    pad: str = "zeros",
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp input with flow."""
    b, _, h, w = feat.size()
    base_grid = mesh_grid(b, h, w).type_as(feat)  # B2HW
    v_grid = norm_grid(base_grid + flow)  # BHW2
    recons = F.grid_sample(
        feat, v_grid, mode=mode, padding_mode=pad, align_corners=align_corners
    )
    return recons


def visualize_mask(
    mask: NDArrayU8,
    colors: Optional[NDArrayU8] = None,
    save_path: Optional[str] = None,
) -> NDArrayU8:
    """Colorize and visualize mask."""
    if colors is None:
        colors = (np.array(np.random.rand(254, 3)) * 255).astype(np.uint8)
    mask_rpt = np.repeat(mask[:, :, None], 3, axis=2)
    vis_mask = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    for inst_id in np.unique(mask_rpt):
        if inst_id == IGNORE_LABEL:
            continue
        num_pix = len(vis_mask[mask_rpt == inst_id]) // 3
        vis_mask[mask_rpt == inst_id] = np.tile(colors[inst_id], num_pix)
    if save_path is not None:
        Image.fromarray(vis_mask).save(save_path)
    return vis_mask  # type: ignore


def convert_rles_to_mask(
    labels: List[Label],
    image_size: ImageSize,
    id_list: Optional[List[str]] = None,
    filter_mask: bool = False,
    filter_size: int = 0,
) -> Tuple[NDArrayU8, List[str]]:
    """Convert list of RLEs to segmentation mask."""
    out_mask = (
        np.ones((image_size.height, image_size.width), dtype=np.uint8)
        * IGNORE_LABEL
    )
    gt_ids = [] if id_list is None else id_list
    for label in labels:
        if label.rle is None:
            continue
        mask = rle_to_mask(label.rle)
        if id_list is None:
            if filter_mask and mask.sum() < filter_size:
                continue
            if label.id not in gt_ids:
                gt_ids.append(label.id)
        if id_list is None or label.id in gt_ids:
            out_mask[mask > 0] = gt_ids.index(label.id)
    return out_mask, gt_ids  # type: ignore


def warp_segmentation_mask(mask: NDArrayU8, flow: NDArrayF64) -> NDArrayU8:
    """Warp segmentation mask with flow."""
    # remap ignore label to 0 since grid sample only has zero padding
    mask = copy.deepcopy(mask)
    mask[mask == 0] = len(np.unique(mask)) - 1
    mask[mask == IGNORE_LABEL] = 0
    mask_tensor = torch.Tensor(mask).unsqueeze(0).unsqueeze(0).float()
    flow_tensor = torch.Tensor(flow).unsqueeze(0).permute(0, 3, 1, 2)
    warp_mask = flow_warp(mask_tensor, flow_tensor, mode="nearest")
    out_mask = warp_mask.squeeze().numpy().astype(np.uint8)
    out_mask[out_mask == 0] = IGNORE_LABEL
    out_mask[out_mask == len(np.unique(out_mask)) - 1] = 0
    return out_mask


def per_class_prec(hist: NDArrayI32) -> NDArrayF64:
    """Calculate per class precision."""
    precs = safe_divide(np.diag(hist), hist.sum(axis=1))
    # Last class as `ignored`
    return precs[:-1]  # type: ignore


def compute_scores_per_frame(
    ann_mask: NDArrayU8, pred_mask: NDArrayU8
) -> Optional[Tuple[float, float, float, float]]:
    """Compute accuracy and IoU of flow prediction."""
    ann_mask = ann_mask[IGNORE_H:-IGNORE_H, IGNORE_W:-IGNORE_W]
    pred_mask = pred_mask[IGNORE_H:-IGNORE_H, IGNORE_W:-IGNORE_W]
    num_gt = len(np.unique(ann_mask)) - 1
    if num_gt == 0:
        return None
    ann_mask = copy.deepcopy(ann_mask)
    ann_mask[ann_mask > num_gt] = num_gt
    hist = fast_hist(ann_mask.flatten(), pred_mask.flatten(), num_gt + 1)
    acc = float(safe_divide(np.diag(hist).sum(), hist[:, :-1].sum()))
    pc_acc, pc_iou = per_class_acc(hist).mean(), per_class_iou(hist).mean()
    prec = per_class_prec(hist).mean()
    return acc, pc_acc, pc_iou, prec


def compute_scores_per_video(
    ann_frames: List[Frame], pred_path: str, image_size: ImageSize
) -> Tuple[float, float, float, float]:
    """Compute flow scores per video."""
    scores_list = []
    for ann_frame1, ann_frame2 in zip(ann_frames[::2], ann_frames[1::2]):
        # ann_frame1, ann_frame2 = ann_frames[idx], ann_frames[idx + step]
        if ann_frame1.labels is None or ann_frame2.labels is None:
            continue
        # gt_ids = [label.id for label in ann_frame1.labels]
        ann_mask1, gt_ids = convert_rles_to_mask(
            ann_frame1.labels, image_size, filter_mask=True
        )
        ann_mask2, _ = convert_rles_to_mask(
            ann_frame2.labels, image_size, gt_ids
        )
        assert ann_frame1.videoName is not None
        flow_name = ann_frame1.name.replace(".jpg", ".flo")
        flow_path = os.path.join(pred_path, ann_frame1.videoName, flow_name)
        if os.path.exists(flow_path):
            flow = read_flow(flow_path)
            warp_mask = warp_segmentation_mask(ann_mask2, flow)
        else:
            print(f"Missing flow for {flow_path}.")
            warp_mask = ann_mask2
        scores = compute_scores_per_frame(ann_mask1, warp_mask)
        # colors = (np.array(np.random.rand(254, 3)) * 255).astype(np.uint8)
        # visualize_mask(warp_mask, colors, 'vis_warp.png')
        # visualize_mask(ann_mask1, colors, 'vis_ann1.png')
        # visualize_mask(ann_mask2, colors, 'vis_ann2.png')
        # print(scores)
        # breakpoint()
        if scores is None:
            continue
        scores_list.append(scores)
    return tuple(
        np.mean([s[i] for s in scores_list]) * 100.0 for i in range(4)
    )


def group_by_video(gt_frames: List[Frame]) -> List[List[Frame]]:
    """Group frames by video."""
    frames_dict = defaultdict(list)
    for frame in gt_frames:
        assert frame.videoName is not None
        frames_dict[frame.videoName].append(frame)
    return [frames_dict[k] for k in frames_dict]


def evaluate_flow(
    gt_frames: List[Frame], pred_path: str, config: Config, nproc: int = NPROC
) -> FlowResult:
    """Evaluate optical flow with Scalabel format.

    Args:
        gt_frames: the ground truth frames.
        pred_path: the prediction path.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        FlowResult: evaluation results.
    """
    img_size = config.imageSize
    assert img_size is not None
    ann_videos = group_by_video(gt_frames)
    # ann_videos = [
    #     f
    #     for f in ann_videos
    #     if f[0].videoName
    #     in [
    #         # "b1d0a191-03dcecc2",
    #         # "b1d22ed6-f1cac061",
    #         "b1c9c847-3bda4659",
    #         "b1d0a191-06deb55d",
    #         "b1d10d08-c35503b8",
    #     ]
    # ]
    if nproc > 1:
        with Pool(nproc) as pool:
            scores = pool.map(
                partial(
                    compute_scores_per_video,
                    pred_path=pred_path,
                    image_size=img_size,
                ),
                tqdm(ann_videos),
            )
    else:
        scores = [
            compute_scores_per_video(ann_frames, pred_path, img_size)
            for ann_frames in tqdm(ann_videos)
        ]
    metrics = ["Accuracy", "mAccuracy", "mIoU", "mPrecision"]
    res_dict = {}
    for i, m in enumerate(metrics):
        res_dict[m] = [{AVERAGE: np.mean([s[i] for s in scores])}]

    return FlowResult(**res_dict)


def setup_frames_by_step(gt_frames: List[Frame], step: int) -> List[Frame]:
    """Convert GT frames to flow eval format."""
    gt_videos = group_and_sort(gt_frames)
    gt_frames_step = []
    for gt_video in gt_videos:
        for idx in range(len(gt_video) - step):
            gt_frames_step.extend([gt_video[idx], gt_video[idx + step]])
    return gt_frames_step


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Boundary evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to boundary ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=False, help="path to flow results"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config toml file. Contains definition of categories, "
        "and optionally attributes and resolution. For an example "
        "see scalabel/label/testcases/configs.toml",
    )
    parser.add_argument(
        "--out-file",
        default="",
        help="Output file for flow evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for flow evaluation",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=1,
        help="step size for flow evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt, args.nproc)
    gts, cfg = dataset.frames, dataset.config
    # preds = load(args.result).frames
    # if args.config is not None:
    #     cfg = load_label_config(args.config)
    cfg = load_bdd100k_config("seg_track").scalabel
    if cfg is None:
        raise ValueError(
            "Dataset config is not specified. Please use --config"
            " to specify a config for this dataset."
        )
    gts = setup_frames_by_step(gts, args.step)
    eval_result = evaluate_flow(gts, args.result, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.json(), fp)
