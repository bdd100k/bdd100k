"""Scripts for running SfM on bdd100k sequences."""
import argparse
import os
import time
import pdb
from typing import List, Optional, Tuple
import numpy as np
from scalabel.common.typing import NDArrayF64
from scalabel.label.typing import Frame, Intrinsics
from scalabel.label.utils import (
    get_extrinsics_from_matrix,
    get_matrix_from_extrinsics,
)
from .colmap.database_io import COLMAPDatabase
from .colmap.model_io import CAMERA_MODEL_NAMES, Camera, Image, rotmat2qvec
from .utils import cam_spec_prior, frames_from_images, get_pose_priors, get_gps_priors

"""Define the location of colmap"""
colmap_ = "/scratch_net/zoidberg/yuthan/software/__install__/bin/colmap"

def add_image_ids(frames: List[Frame], db_path: str) -> None:
    """Add the image ids in the database to each frame (map by name)."""
    db = COLMAPDatabase.connect(db_path)
    rows = db.execute("SELECT * FROM images")
    for row in rows:
        img_id, name = row[0], row[1]
        for f in frames:
            if f.name == name:
                f.attributes = {"id": img_id}
    db.close()

def add_spatials_to_db(frames: List[Frame], db_path: str) -> None:
    """Add the spatial locations (transformed to cartesian) to images in database."""
    db = COLMAPDatabase.connect(db_path)
    for f in frames:
        qvec, tvec = [], []
        if f.extrinsics is not None and f.attributes is not None:
            trans_mat = get_matrix_from_extrinsics(f.extrinsics)
            qvec = rotmat2qvec(trans_mat[:3, :3])
            tvec = trans_mat[:3, 3]
            [qw, qx, qy, qz] = qvec
            [tx, ty, tz] = tvec
            id = f.attributes["id"]
            db.execute(
                """
                UPDATE images 
                SET prior_tx=%s, prior_ty=%s, prior_tz=%s
                Where image_id=%s"""
                % (tx, ty, tz, id)
            )
    # Uncomment to print table content in database
    # rows = db.execute("SELECT * FROM images")
    # for row in rows:
    #     print(row)
    db.commit()
    db.close()

def add_gps_to_db(frames: List[Frame], db_path: str) -> None:
    """Add the gps coordinates locations to images in database."""
    db = COLMAPDatabase.connect(db_path)
    for f in frames:
        if f.extrinsics is not None and f.attributes is not None:
            tx = f.extrinsics.location[0]
            ty = f.extrinsics.location[1]
            tz = 0
            id = f.attributes["id"]
            db.execute(
                """
                UPDATE images 
                SET prior_tx=%s, prior_ty=%s, prior_tz=%s
                Where image_id=%s"""
                % (tx, ty, tz, id)
            )
    # Uncomment to print table content in database
    # rows = db.execute("SELECT * FROM images")
    # for row in rows:
    #     print(row)
    db.commit()
    db.close()

def feature_extractor(
    frames: List[Frame],
    image_path: str,
    output_path: str,
    info_path: Optional[str] = None
) -> List[Frame]:
    """Conduct feature extraction."""
    # we assume shared intrinsics
    if frames[0].intrinsics is not None:
        assert frames[0].intrinsics.focal[0] == frames[0].intrinsics.focal[0]
        intrinsics = frames[0].intrinsics
    else:
        intrinsics = cam_spec_prior()

    f_x = intrinsics.focal[0]
    c_x, c_y = intrinsics.center

    options = (
        " --ImageReader.camera_model SIMPLE_PINHOLE"
        " --ImageReader.single_camera 1"
        f" --ImageReader.camera_params {f_x},{c_y},{c_x}"
    )

    os.system(
        f"colmap feature_extractor --database_path {output_path}/database.db "
        f"--image_path {image_path}" + options
    )

    return frames


def database_creator(output_path: str) -> None:   
    os.system(
        f"{colmap_} database_creator --database_path {output_path}/database.db"
    )


def feature_matcher(
    frames: List[Frame],
    matcher_method: str,
    image_path: str,
    output_path: str,
    info_path: Optional[str] = None,
    num_neighbors: Optional[int] = 50,
) -> List[Frame]:
    """Conduct feature matcher."""
    # we assume shared intrinsics
    if frames[0].intrinsics is not None:
        assert frames[0].intrinsics.focal[0] == frames[0].intrinsics.focal[0]
        intrinsics = frames[0].intrinsics
    else:
        intrinsics = cam_spec_prior()

    add_image_ids(frames, f"{output_path}/database.db")
    add_gps_to_db(frames, f"{output_path}/database.db")

    if matcher_method == "exhaustive":  
        os.system(
            f"colmap exhaustive_matcher --database_path {output_path}/database.db "
            f"--SiftMatching.min_inlier_ratio {0.65} "
            f"--SiftMatching.max_distance {0.55} "
            f"--SiftMatching.min_num_inliers {50}"
        )
    elif matcher_method == "spatial":  
        os.system(
            f"colmap spatial_matcher --database_path {output_path}/database.db "
            f"--SiftMatching.min_inlier_ratio {0.25} "
            f"--SiftMatching.min_num_inliers {15} " 
            "--SpatialMatching.is_gps 1 "
            f"--SpatialMatching.max_num_neighbors {num_neighbors} "
            "--SpatialMatching.max_distance 100"
        )
    elif matcher_method == "sequential":  
        os.system(
            f"colmap sequential_matcher --database_path {output_path}/database.db "
            f"--SiftMatching.min_inlier_ratio {0.65} "
            f"--SiftMatching.max_distance {0.55} "
            f"--SiftMatching.min_num_inliers {60}"
        )
    return frames


def mapper(
    image_path: str,
    output_path: str,
    sparse_path: str,
) -> List[Frame]: 
    """Conduct incremental mapper."""
    options = (
        " --Mapper.ba_refine_principal_point 1"
        " --Mapper.min_num_matches 5"
        )
    os.system(
        f"colmap mapper "
        f"--database_path {output_path}/database.db "
        f"--image_path {image_path} "
        f"--output_path {sparse_path} "
        f"--Mapper.abs_pose_min_inlier_ratio {0.25} "
        f"--Mapper.filter_max_reproj_error {4.0} "
        f"--Mapper.min_num_matches {20}"
    )


def new_mapper(
    image_path: str,
    output_path: str,
    sparse_path: str,
):   
    """Conduct incremental mapper using the modified colmap."""
    options = (
        " --Mapper.ba_refine_principal_point 1"
        )
    os.system(
        f"{colmap_} mapper "
        f"--database_path {output_path}/database.db "
        f"--image_path {image_path} "
        f"--output_path {sparse_path} "
        f"--Mapper.abs_pose_min_inlier_ratio {0.25} "
        f"--Mapper.filter_max_reproj_error {4.0} "
        f"--Mapper.min_num_matches {20} "
        f"--Mapper.prior_is_gps 1 "
        f"--Mapper.use_prior_motion 1 "
        f"--Mapper.use_enu_coords 1 "
        f"--Mapper.prior_loss_scale 0.072 "
        f"--Mapper.ba_global_loss_scale 10.597 "
    )  

def orientation_aligner(
    image_path: str,
    project_path: str,
    sparse_path: str,
    output_path: str,
) -> List[Frame]:   
    os.system(
        f"colmap model_orientation_aligner "
        f"--image_path {image_path} "
        f"--project_path {project_path} "
        f"--input_path {sparse_path} "
        f"--output_path {output_path} "
    )


def create_info_path(num_seqs, seq_names):
    if num_seqs == 1:
        info_path = "/srv/beegfs02/scratch/bdd100k/data/bdd100k/info/" + seq_names[0]
    else:
        info_path = []
        for seq_name in seq_names:
            info_path.append("/srv/beegfs02/scratch/bdd100k/data/bdd100k/info/" + seq_name)
    return info_path


def main():
    parser = argparse.ArgumentParser(description="Sparse Reconstruction for a sequence")
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        help="Path to image sequence.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Path to output (the path contains database, sparse, dense).",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default=None,
        help="Path to info file in .json for sequence.",
    )
    parser.add_argument(
        "--matcher-method",
        type=str,
        help="The method for feature matcher. (spatial, sequential, exhaustive).",
    )
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        default="feature",
        help="Which job to do(feature, mapper, or both)",
    )

    args = parser.parse_args()
    sparse_path = os.path.join(args.output_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)

    frames = get_gps_priors(args.info_path, args.image_path)
    if frames is None:
        frames = frames_from_images(args.image_path)
    

    if args.job == "feature":
        database_creator(args.output_path)
        while not os.path.exists(f"{args.output_path}/database.db"):
            time.sleep(0.5)
        frames = feature_extractor(frames, args.image_path, args.output_path, args.info_path)
        frames = feature_matcher(frames, args.matcher_method, args.image_path, args.output_path, args.info_path)
    elif args.job == "mapper":
        new_mapper(args.image_path, args.output_path, sparse_path)   
    elif args.job == "both":
        database_creator(args.output_path)
        while not os.path.exists(f"{args.output_path}/database.db"):
            time.sleep(0.5)
        frames = feature_extractor(frames, args.image_path, args.output_path, args.info_path)
        frames = feature_matcher(frames, args.matcher_method, args.image_path, args.output_path, args.info_path)
        new_mapper(image_path, agrs.output_path, sparse_path) 

if __name__ == "__main__":
    main()