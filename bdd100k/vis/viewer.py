"""An offline label visualizer for BDD100K file.

Works for 2D / 3D bounding box, segmentation masks, etc.
"""

import argparse
import concurrent.futures
from typing import Dict

import numpy as np
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64
from scalabel.label.typing import Label
from scalabel.vis.controller import (
    ControllerConfig,
    DisplayConfig,
    ViewController,
)
from scalabel.vis.label import LabelViewer, UIConfig

from ..label.label import drivables, labels, lane_categories


class LabelViewerBDD100K(LabelViewer):
    """Basic class for viewing BDD100K labels."""

    def __init__(self, ui_cfg: UIConfig) -> None:
        """Initializer."""
        super().__init__(ui_cfg)
        self.colors: Dict[str, NDArrayF64] = {
            label.name: np.array(label.color)
            for label in labels
            if not label.hasInstances
        }
        self.colors.update(
            {drivable.name: np.array(drivable.color) for drivable in drivables}
        )
        self.colors.update(
            {lane.name: np.array(lane.color) for lane in lane_categories}
        )

    def _get_label_color(self, label: Label) -> NDArrayF64:
        """Get color by category and id."""
        if label.category in self.colors:
            return self.colors[label.category] / 255.0
        return super()._get_label_color(label)


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser(
        """
Interface keymap:
    -  n / p: Show next or previous image
    -  Space: Start / stop animation
    -  t: Toggle 2D / 3D bounding box (if avaliable)
    -  a: Toggle the display of the attribute tags on boxes or polygons.
    -  c: Toggle the display of polygon vertices.
    -  Up: Increase the size of polygon vertices.
    -  Down: Decrease the size of polygon vertices.
Export images:
    - add `-o {dir}` tag when runing.
    """
    )
    parser.add_argument("-i", "--image-dir", help="image directory")
    parser.add_argument(
        "-l",
        "--labels",
        required=False,
        default="labels.json",
        help="Path to the json file",
        type=str,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the image (px)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the image (px)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Scale up factor for annotation factor. "
        "Useful when producing visualization as thumbnails.",
    )
    parser.add_argument(
        "--no-attr",
        action="store_true",
        default=False,
        help="Do not show attributes",
    )
    parser.add_argument(
        "--no-box3d",
        action="store_true",
        default=True,
        help="Do not show 3D bounding boxes",
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        default=False,
        help="Do not show tags on boxes or polygons",
    )
    parser.add_argument(
        "--no-vertices",
        action="store_true",
        default=False,
        help="Do not show vertices",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        type=str,
        help="output image directory with label visualization. "
        "If it is set, the images will be written to the "
        "output folder instead of being displayed "
        "interactively.",
    )
    parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="from which frame to visualize. Default is 0.",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        default=-1,
        help="up to which frame to visualize. Default is -1, "
        "indicating loading all frames for visualizatoin.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for json loading",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    # Initialize the thread executor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        ui_cfg = UIConfig(
            height=args.height,
            width=args.width,
            scale=args.scale,
        )
        display_cfg = DisplayConfig(
            with_attr=not args.no_attr,
            with_box2d=args.no_box3d,
            with_box3d=not args.no_box3d,
            with_ctrl_points=not args.no_vertices,
            with_tags=not args.no_tags,
        )
        viewer = LabelViewer(ui_cfg)

        ctrl_cfg = ControllerConfig(
            image_dir=args.image_dir,
            label_path=args.labels,
            out_dir=args.output_dir,
            nproc=args.nproc,
            range_begin=args.range_begin,
            range_end=args.range_end,
        )
        controller = ViewController(ctrl_cfg, display_cfg, executor)
        viewer.run_with_controller(controller)


if __name__ == "__main__":
    main()
