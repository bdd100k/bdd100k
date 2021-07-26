"""An offline label visualizer for BDD100K file.

Works for 2D / 3D bounding box, segmentation masks, etc.
"""

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from PIL import Image
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64, NDArrayU8
from scalabel.label.typing import Label
from scalabel.vis.controller import ControllerConfig, ViewController
from scalabel.vis.viewer import DisplayConfig, LabelViewer, UIConfig

from ..label.label import drivables, labels, lane_categories


class LabelViewerBDD100K(LabelViewer):
    """Basic class for viewing BDD100K labels."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        """Initializer."""
        super().__init__(*args, **kwargs)
        self.colors = {
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
            return self.colors[label.category]
        return super()._get_label_color(label)


@dataclass
class ControllerConfigBDD100K(ControllerConfig):
    """Visulizer's config class for BDD100K."""

    color_dir: str

    def __init__(  # type: ignore
        self, color_dir: str, *args, **kwargs
    ) -> None:
        """Initialize with args."""
        super().__init__(*args, **kwargs)
        self.color_dir = color_dir


class ViewControllerBDD100K(ViewController):
    """Visualization controller for BDD100K."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        """Initializer."""
        super().__init__(*args, **kwargs)
        self.config_bdd100k = cast(ControllerConfigBDD100K, self.config)

    def show_frame(self) -> bool:
        """Show one frame in matplotlib axes."""
        plt.cla()
        frame = self.frames[self.frame_index % len(self.frames)]
        # Fetch the image
        img: NDArrayU8 = self.images[frame.name].result()

        if self.config_bdd100k.color_dir is not None:
            image_seg_path = os.path.join(
                self.config_bdd100k.color_dir, frame.name.replace("jpg", "png")
            )
            if os.path.exists(image_seg_path):
                print("Local segmentation image path:", image_seg_path)
                colormap: NDArrayU8 = np.asarray(
                    Image.open(image_seg_path)
                ).astype(np.uint8)
                mixed = (
                    np.multiply(img, 0.5) + np.multiply(colormap, 0.5)
                ).astype(np.uint8)
                self.viewer.draw_image(mixed, frame.name)
                return True
            print("Colormap not found.")

        self.viewer.draw_image(img, frame.name)

        # show label
        if frame.labels is None or len(frame.labels) == 0:
            print("No labels found")
            return True

        _labels = frame.labels
        if self.config.with_attr:
            self.viewer.draw_attributes(frame)
        if self.config.with_box2d:
            self.viewer.draw_box2ds(_labels)
        if self.config.with_box3d and frame.intrinsics is not None:
            self.viewer.draw_box3ds(_labels, frame.intrinsics)
        if self.config.with_poly2d:
            self.viewer.draw_poly2ds(_labels)

        return True


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
        "-c",
        "--color-dir",
        required=False,
        default=None,
        type=str,
        help="Path to the colormap directory. If it is set, the image will be"
        "mixed with the corresponding colormap.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Scale up factor for annotation factor. "
        "Useful when producing visualization as "
        "thumbnails.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the image (px)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the image (px)",
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
        "--output_dir",
        required=False,
        default=None,
        type=str,
        help="output image directory with label visualization. "
        "If it is set, the images will be written to the "
        "output folder instead of being displayed "
        "interactively.",
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
            show_ctrl_points=not args.no_vertices,
            show_tags=not args.no_tags,
            ctrl_points_size=2.0,
        )
        ctrl_cfg = ControllerConfigBDD100K(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            color_dir=args.color_dir,
            with_mask=not args.color_dir,
        )
        viewer = LabelViewerBDD100K(ui_cfg, display_cfg)
        controller = ViewController(
            ctrl_cfg, viewer, args.labels, args.nproc, executor
        )
        controller.view()


if __name__ == "__main__":
    main()
