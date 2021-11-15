"""BDD100K SfM demo."""
import subprocess
from typing import List

from .run import frames_to_colmap
from .utils import frames_from_images, interpolate_trajectory, load_pose_data


def main(sequences: List[str]) -> None:
    """Demo SfM reconstruction per sequence."""
    for seq in sequences:
        prepare_data = f"""
        mkdir {seq}
        cd {seq}
        wget http://dl.yf.io/bdd100k/images/all_zip/{seq}.zip
        unzip {seq}.zip && mv {seq} images
        mkdir colmap_reconstruction
        """
        subprocess.check_output(prepare_data, shell=True)
        # TODO upload infos
        info_path = (
            f"/srv/beegfs02/scratch/bdd100k/data/bdd100k/info/{seq}.json"
        )

        # get gps pose priors from info
        pose_priors = load_pose_data(info_path)
        assert pose_priors is not None
        frames = frames_from_images(f"{seq}/images/")
        interpolate_trajectory(pose_priors, frames)

        frames_to_colmap(frames, f"{seq}/prior/")

        # TODO execute colmap, visualize result


if __name__ == "__main__":
    demo_seq = ["0000f77c-6257be58"]
    main(demo_seq)
