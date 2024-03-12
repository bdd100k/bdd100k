"""Scripts for getting where to download depth and poses for a sequence."""
import argparse
import os

from tqdm import tqdm

try:
    import pickle
except ImportError:
    pass
from bdd100k.sfm.colmap.database_io import COLMAPDatabase  # type: ignore
from bdd100k.sfm.colmap.read_write_model import (  # type: ignore
    Image,
    read_images_binary,
)


class Frame:
    """A class represents a frame in sequence.

    Args:
        image: an image object from colmap
    """

    def __init__(self, image: Image):
        """Create a frame from image."""
        self.name = image.name
        self.tvec = image.tvec
        self.qvec = image.qvec
        self.depth_path = ""
        self.image_path = ""
        self.gps = ()

    def get_depth_path(self) -> str:
        """Return the path of the depth."""
        return self.depth_path


class Postcode:
    """A class represents a postcode of data.

    Args:
        postcode_name: name of postcode, e.g. 10001
        data_root: path where data is saved
    """

    def __init__(self, postcode_name: str, data_root: str):
        """Create a postcode object."""
        self.postcode_name = postcode_name
        self.postcode_path = os.path.join(data_root, postcode_name)
        self.overlaps = {}  # type: ignore
        self.singles = {}  # type: ignore

    def get_name(self) -> str:
        """Return the name of postcode."""
        return self.postcode_name


class Scene:
    """A class represents a scene, either an overlap or a single.

    Args:
        postcode: Postcode
        scene_name: name of scene
    """

    def __init__(self, postcode: Postcode, scene_name: str):
        """Create a Scene object."""
        self.postcode = postcode
        self.scene_name = scene_name
        self.scene_type = ""
        if len(scene_name) == 17:
            self.scene_type = "singles"
        else:
            self.scene_type = "overlaps"
        self.scene_path = os.path.join(
            postcode.postcode_path, "daytime", self.scene_type, scene_name
        )
        self.sequences = {}  # type: ignore
        self.fused_pcd_path = []  # type: ignore

    def get_sequences(self) -> None:
        """Get all sequence information from the scene."""
        denses_path = os.path.join(self.scene_path, "dense")
        database_path = os.path.join(self.scene_path, "database.db")
        db = COLMAPDatabase.connect(database_path)
        rows = db.execute("SELECT * FROM images")
        db_dict = {}
        for row in rows:
            img_name, lat_, long_ = row[1], row[7], row[8]
            db_dict[img_name] = (lat_, long_)
        for dense_path in os.listdir(denses_path):
            sparse_path = os.path.join(denses_path, dense_path, "sparse")
            if sparse_path == "":
                print(f"The {sparse_path} is missing sparse information")
                return
            colmap_images = read_images_binary(
                os.path.join(sparse_path, "images.bin")
            )
            images_path = os.path.join(self.scene_path, "images")
            depth_maps_path = os.path.join(
                denses_path,
                dense_path,
                "depth_maps_nbr_consistent",
            )
            fused_pcd_path = os.path.join(denses_path, dense_path, "fused.ply")
            for image in colmap_images.values():
                sequence_name = image.name[:17]
                frame = Frame(image)
                depth_name = image.name.replace("jpg", "png")
                depth_path = os.path.join(depth_maps_path, depth_name)
                image_path = os.path.join(images_path, image.name)
                frame.depth_path = depth_path
                frame.image_path = image_path
                frame.gps = db_dict[image.name]  # type: ignore

                if sequence_name in self.sequences:
                    self.sequences[sequence_name].add_frame(frame)
                else:
                    sequence = Sequence(sequence_name, self)
                    self.sequences[sequence_name] = sequence
                    self.sequences[sequence_name].add_frame(frame)
            self.fused_pcd_path.append(fused_pcd_path)


class Sequence:
    """A class represents a sequence.

    Args:
        sequence_name: name of sequence
        scene: the scene the sequence belongs to
    """

    def __init__(self, sequence_name: str, scene: Scene):
        """Create a Sequence."""
        self.sequence_name = sequence_name
        self.scene = scene
        self.frames = {}  # type: ignore

    def add_frame(self, frame: Frame) -> None:
        """Add a frame to the sequence."""
        self.frames[frame.name] = frame


class BDD100KDepthDataset:
    """Dataset that returns depth and poses information of a bdd100k sequence.

    Args:
        data_root: base path where the data are downloaded
        sequence: sequence name
    """

    def __init__(self, data_root: str):
        """Initizalize the scene information."""
        super().__init__()
        self.data_root = data_root
        self.postcodes = {}  # type: ignore
        self.sequences = {}  # type: ignore

    def get_dataset(self) -> None:
        """Locate the sequence in the dataset and extract information."""
        for postcode_name in tqdm(os.listdir(self.data_root)):
            print(f"Getting data from postcode {postcode_name} ...")
            postcode_path = os.path.join(self.data_root, postcode_name)
            if not os.path.isdir(postcode_path):
                continue
            postcode = Postcode(postcode_name, self.data_root)
            self.postcodes[postcode_name] = postcode

            # A scene could be either a single or overlap
            singles_path = os.path.join(postcode_path, "daytime", "singles")
            overlaps_path = os.path.join(postcode_path, "daytime", "overlaps")

            if os.path.exists(singles_path):
                for single_name in os.listdir(singles_path):
                    scene = Scene(postcode, single_name)
                    scene.get_sequences()
                    self.sequences = {**self.sequences, **scene.sequences}
            if os.path.exists(overlaps_path):
                for overlap_name in os.listdir(overlaps_path):
                    scene = Scene(postcode, overlap_name)
                    scene.get_sequences()
                    self.sequences = {**self.sequences, **scene.sequences}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get information from bdd100k depth."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root path of the downloaded data",
    )
    ARGUMENTS = parser.parse_args()

    dataset_summary_path = os.path.join(
        ARGUMENTS.data_root, "bdd100k_depth_dataset.pkl"
    )
    if os.path.exists(dataset_summary_path):
        with open(dataset_summary_path, "rb") as f:
            bdd100k_depth_dataset = pickle.load(f)
    else:
        bdd100k_depth_dataset = BDD100KDepthDataset(ARGUMENTS.data_root)
        bdd100k_depth_dataset.get_dataset()
        with open(dataset_summary_path, "wb") as f:  # type: ignore
            pickle.dump(bdd100k_depth_dataset, f)

    for seq_name, seq in bdd100k_depth_dataset.sequences.items():
        cur_scene = seq.scene
        print(f"Sequence {seq_name} is a {cur_scene.scene_type}")
        # If the scene is an overlap, the overlapped sequences are:
        overlapped_seqs = cur_scene.sequences
        for cur_frame in seq.frames.values():
            cur_image_path = cur_frame.image_path
            cur_depth_path = cur_frame.depth_path
            cur_tvec = cur_frame.tvec
            cur_qvec = cur_frame.qvec
            cur_gps = cur_frame.gps
            # print(
            #     f"Image path is: {cur_image_path} \n"
            #     f"Depth path is: {cur_depth_path} \n"
            #     f"Pose tvec (colmap) is: {cur_tvec} \n"
            #     f"Pose qvec (colmap) is: {cur_qvec} \n"
            #     f"GPS is: {cur_gps}")
