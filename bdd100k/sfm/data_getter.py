"""Scripts for getting where to download depth and poses for a sequence."""
import argparse
import glob
import json
import os

try:
    import cv2
except ImportError:
    pass
from bdd100k.sfm.colmap.read_write_model import (  # type: ignore
    read_images_binary,
)


class BDD100KDepthDataset:
    """Dataset that returns depth and poses information of a bdd100k sequence.

    Args:
        data_root: base path where the data are downloaded
        sequence: sequence name
    """

    def __init__(self, data_root: str, sequence: str):
        """Initizalize the scene information."""
        super().__init__()
        self.data_root = data_root
        self.sequence = sequence
        self.scene_type = ""
        self.postcode = ""
        self.scene_name = ""
        self.depth_path = ""
        self.sparse_path = ""
        self.fused_pcd_path = ""
        self.images = {}  # type: ignore
        self.get_scenes()

    def get_scenes(self) -> None:
        """Locate the sequence in the dataset and extract information."""
        for postcode in os.listdir(self.data_root):
            postcode_path = os.path.join(self.data_root, postcode)
            if not os.path.isdir(postcode_path):
                continue
            for scene_type in ["singles", "overlaps"]:
                scene_path = os.path.join(postcode_path, "daytime", scene_type)
                if not os.path.exists(scene_path):
                    continue
                for scene in os.listdir(scene_path):
                    denses_path = os.path.join(scene_path, scene, "dense")
                    images_path = os.path.join(scene_path, scene, "images")
                    for dense_path in os.listdir(denses_path):
                        depth_maps_path = os.path.join(
                            denses_path,
                            dense_path,
                            "depth_maps_nbr_consistent",
                        )
                        sparse_path = os.path.join(
                            denses_path, dense_path, "sparse"
                        )
                        fused_pcd_path = os.path.join(
                            denses_path, dense_path, "fused.ply"
                        )
                        files = glob.glob(
                            depth_maps_path + "/" + self.sequence + "*"
                        )
                        if files:
                            self.scene_type = scene_type
                            self.scene_name = scene
                            self.images_path = images_path
                            self.postcode = postcode
                            self.depth_path = depth_maps_path
                            self.sparse_path = sparse_path
                            self.fused_pcd_path = fused_pcd_path
                        else:
                            break

    def get_images(self) -> None:
        """Update self.images, including image name and poses."""
        # return frames within recon
        if self.sparse_path == "":
            print("The sequence is missing sparse information")
            return
        colmap_images = read_images_binary(
            os.path.join(self.sparse_path, "images.bin")
        )
        for image in colmap_images.values():
            if image.name[:17] == self.sequence:
                depth_name = image.name.replace("jpg", "png")
                self.images[image.name] = {
                    "tvec": image.tvec.tolist(),
                    "qvec": image.qvec.tolist(),
                    "image_path": os.path.join(self.images_path, image.name),
                    "depth_path": os.path.join(self.depth_path, depth_name),
                }

    def get_depth(self) -> None:
        """Update depth information in self.images."""
        if not self.images:
            print("Images are missing")
            return
        for name in self.images.keys():
            cur_depth_name = name.replace("jpg", "png")
            cur_depth_path = os.path.join(self.depth_path, cur_depth_name)
            if os.path.exists(cur_depth_path):
                cur_depth = (
                    cv2.imread(cur_depth_path, cv2.IMREAD_ANYDEPTH) / 256
                )
                self.images[name]["depth"] = cur_depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get information from bdd100k depth "
    )
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        default="7ab438bb-9ead8b02",  # This is a default example
        help="Which sequence information do you want to get",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root path of the downloaded data",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Whether to save dainformationta into a json",
    )
    ARGUMENTS = parser.parse_args()

    bdd100k_depth_dataset = BDD100KDepthDataset(
        ARGUMENTS.data_root, ARGUMENTS.sequence
    )
    bdd100k_depth_dataset.get_images()

    if ARGUMENTS.save_json:
        save_path = os.path.join(
            ARGUMENTS.data_root, f"{ARGUMENTS.sequence}.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(bdd100k_depth_dataset.__dict__, f, indent=2)
