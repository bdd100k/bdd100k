"""Definition of the BDD100K instance segmentation dataset."""

import json
import os
import os.path as osp
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import pycocotools.mask as mask_utils
from mmdet.datasets import DATASETS
from PIL import Image
from scalabel.common.typing import DictStrAny
from scalabel.label.coco_typing import RLEType
from scalabel.label.io import save
from scalabel.label.typing import Box2D, Frame, Label
from tqdm import tqdm

from bdd100k.eval.ins_seg import evaluate_ins_seg

from .det import BDD100KDetDataset
from .typing import AnnInfo

SHAPE = [720, 1280]


def mask_merge(
    img_name: str,
    scores: List[float],
    segms: List[np.ndarray],
    colors: List[List[int]],
    bitmask_base: str,
) -> None:
    """Merge masks into a bitmask png file."""
    bitmask = np.zeros((*SHAPE, 4), dtype=np.uint8)
    sorted_idxs = np.argsort(scores)
    for idx in sorted_idxs:
        mask = mask_utils.decode(segms[idx])
        for i in range(4):
            bitmask[..., i] = (
                bitmask[..., i] * (1 - mask) + mask * colors[idx][i]
            )
    bitmask_path = osp.join(bitmask_base, img_name.replace(".jpg", ".png"))
    bitmask_dir = osp.split(bitmask_path)[0]
    if not osp.exists(bitmask_dir):
        os.makedirs(bitmask_dir)
    bitmask_img = Image.fromarray(bitmask)
    bitmask_img.save(bitmask_path)


def mask_merge_parallel(
    bitmask_base: str,
    img_names: List[str],
    scores_list: List[List[float]],
    segms_list: List[List[RLEType]],
    colors_list: List[List[List[int]]],
    nproc: int = 4,
) -> None:
    """Merge masks into a bitmask png file. Run parallely."""
    with Pool(nproc) as pool:
        print("\nMerging overlapped masks.")
        pool.starmap(
            partial(mask_merge, bitmask_base=bitmask_base),
            tqdm(
                zip(img_names, scores_list, segms_list, colors_list),
                total=len(img_names),
            ),
        )


@DATASETS.register_module()
class BDD100KInsSegDataset(BDD100KDetDataset):  # type: ignore
    """BDD100K Dataset for instance segmentatio."""

    def __init__(self, bitmask_prefix: str, *args, **kwargs) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        self.bitmask_prefix = bitmask_prefix

    def get_ann_info(self, idx: int) -> AnnInfo:
        """Get annotation information by index."""
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = AnnInfo(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore,
        )

        return ann

    def pre_pipeline(self, results: DictStrAny) -> None:
        """Prepare results dict for pipeline."""
        super().pre_pipeline(results)
        results["bitmask_prefix"] = self.bitmask_prefix

    def format_results(  # pylint: disable=arguments-differ
        self,
        results: List[Tuple[List[np.ndarray], List[List[RLEType]]]],
        out_dir: str,
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results and dataset are not equal: {} != {}".format(
            len(results), len(self)
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        scores_list, segms_list, colors_list = [], [], []
        det_frames, seg_frames = [], []
        img_names = []
        ann_id = 0

        for img_idx in range(len(self)):
            index = 0
            img_name = self.data_infos[img_idx]["file_name"]
            img_names.append(img_name)
            det_frame = Frame(name=img_name, labels=[])
            det_frames.append(det_frame)
            seg_frame = Frame(name=img_name, labels=[])
            seg_frames.append(seg_frame)
            scores, segms, colors = [], [], []

            det_results, seg_results = results[img_idx]
            for cat_idx, [cur_det, cur_seg] in enumerate(
                zip(det_results, seg_results)
            ):
                for bbox, segm in zip(cur_det, cur_seg):
                    ann_id += 1
                    index += 1
                    score = bbox[-1]

                    box2d = Box2D(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]
                    )
                    det_label = Label(
                        id=str(ann_id),
                        score=score,
                        box2d=box2d,
                        category=self.CLASSES[cat_idx],
                    )
                    det_frame.labels.append(det_label)  # type: ignore

                    seg_label = Label(id=str(ann_id), index=index, score=score)
                    seg_frame.labels.append(seg_label)  # type: ignore

                    scores.append(score)
                    segms.append(segm)
                    colors.append([cat_idx + 1, 0, index >> 8, index & 255])

            scores_list.append(scores)
            segms_list.append(segms)
            colors_list.append(colors)

        det_out_path = osp.join(out_dir, "det.json")
        save(det_out_path, det_frames)

        seg_out_path = osp.join(out_dir, "ins_seg.json")
        seg_frame_dicts = [seg_frame.dict() for seg_frame in seg_frames]
        with open(seg_out_path, "w") as fp:
            json.dump(seg_frame_dicts, fp, indent=2)

        bitmask_dir = osp.join(out_dir, "bitmasks")
        mask_merge_parallel(
            bitmask_dir,
            img_names,
            scores_list,
            segms_list,
            colors_list,
            nproc=4,
        )

    def evaluate(  # pylint: disable=arguments-differ
        self,
        results: List[Tuple[List[np.ndarray], List[List[RLEType]]]],
        out_dir: str,
    ) -> Dict[str, float]:
        """Evaluation in COCO protocol with BDD100K bitmasks."""
        self.format_results(results, out_dir)
        pred_base = osp.join(out_dir, "bitmasks")
        score_file = osp.join(out_dir, "ins_seg.json")
        return evaluate_ins_seg(
            self.seg_prefix,
            pred_base,
            score_file,
            self.config.scalabel,
            out_dir,
            self.nproc,
        )
