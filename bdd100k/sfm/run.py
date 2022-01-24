"""Scripts for running SfM on bdd100k sequences."""
import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from scalabel.common.typing import NDArrayF64
from scalabel.label.typing import Frame, Intrinsics
from scalabel.label.utils import (
    get_extrinsics_from_matrix,
    get_matrix_from_extrinsics,
)

from bdd100k.sfm.colmap.model_io import read_model, write_model

from .colmap.database_io import COLMAPDatabase
from .colmap.model_io import CAMERA_MODEL_NAMES, Camera, Image, rotmat2qvec
from .colmap.visualization import Model
from .utils import cam_spec_prior, frames_from_images, get_pose_priors


def frames_to_colmap(
    frames: List[Frame], output_model: str, output_format: str = ".txt"
) -> None:
    """Convert frame sequence to colmap model."""
    f = frames[0]
    cameras, images = {}, {}
    if f.intrinsics is not None:
        assert (
            f.intrinsics.focal[0] == f.intrinsics.focal[1]
        ), "We assume fx = fy in intrinsics."
        width = 1280 if f.size is None else f.size.width
        height = 720 if f.size is None else f.size.height
        camera = Camera(
            id=0,
            model=CAMERA_MODEL_NAMES["SIMPLE_PINHOLE"],
            width=width,
            height=height,
            params=[f.intrinsics.focal[0], *f.intrinsics.center],
        )
        cameras["cam"] = camera

    for f in frames:
        qvec, tvec = [], []
        if f.extrinsics is not None:
            trans_mat = get_matrix_from_extrinsics(f.extrinsics)
            qvec = rotmat2qvec(trans_mat[:3, :3])
            tvec = trans_mat[:3, 3]
        assert f.attributes is not None
        image = Image(
            id=f.attributes["id"],
            camera_id=0,
            name=f.name,
            qvec=qvec,
            tvec=tvec,
            xys=[],
            point3D_ids=[],
        )
        images[f.frameIndex] = image
    os.makedirs(output_model, exist_ok=True)
    write_model(cameras, images, {}, path=output_model, ext=output_format)


def get_frame_index(frames: List[Frame], name: str) -> int:
    """Get index of frame in list by name."""
    for i, f in enumerate(frames):
        if f.name == name:
            return i
    raise ValueError(f"Frame {name} not found in input frames!")


def poses_from_colmap(
    frames: List[Frame], input_model: str, input_format: str = ".txt"
) -> List[Frame]:
    """Read poses from colmap reconstruction, add to frames."""
    cameras, images, pc = read_model(path=input_model, ext=input_format)
    assert len(cameras) == 1, "Camera params should be shared across a video."
    f, c_x, c_y = cameras[0].params
    intrinsics = Intrinsics(focal=(f, f), center=(c_x, c_y))

    for img in images:
        i = get_frame_index(frames, img.name)
        frames[i].intrinsics = intrinsics
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = img.qvec2rotmat()
        trans_mat[:3, 3] = img.tvec
        frames[i].extrinsics = get_extrinsics_from_matrix(trans_mat)
    return frames


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
    """Add the spatial locations to images in database."""
    db = COLMAPDatabase.connect(db_path)

    for f in frames:
        qvec, tvec = [], []
        if f.extrinsics is not None:
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


def sfm_reconstruction(
    image_path: str,
    output_path: str,
    info_path: Optional[str] = None,
    output_format: str = ".txt",
    cycles: int = 2,
) -> Tuple[List[Frame], NDArrayF64]:
    """Execute SfM reconstruction, return posed frames and 3D points."""
    has_pose_prior = False
    if info_path is not None:
        frames = get_pose_priors(info_path, image_path)
        if frames is None:
            frames = frames_from_images(image_path)
        else:
            has_pose_prior = True
    else:
        frames = frames_from_images(image_path)

    os.makedirs(output_path, exist_ok=True)
    os.system(
        f"colmap database_creator --database_path {output_path}/database.db"
    )

    while not os.path.exists(f"{output_path}/database.db"):
        time.sleep(0.5)

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
        f" --ImageReader.camera_params {f_x},{c_x},{c_y}"
    )

    os.system(
        f"colmap feature_extractor --database_path {output_path}/database.db "
        f"--image_path {image_path}" + options
    )

    add_image_ids(frames, f"{output_path}/database.db")
    add_spatials_to_db(frames, f"{output_path}/database.db")

    # os.system(
    #     f"colmap sequential_matcher --database_path {output_path}/database.db "
    #     f"--SiftMatching.min_inlier_ratio {0.65} "
    #     f"--SiftMatching.max_distance {0.55} "
    #     f"--SiftMatching.min_num_inliers {60}"
    # )

    os.system(
        f"colmap spatial_matcher --database_path {output_path}/database.db "
        f"--SiftMatching.min_inlier_ratio {0.75} "
        f"--SiftMatching.max_distance {0.45} "
        f"--SiftMatching.min_num_inliers {70} "
        "--SpatialMatching.is_gps 0 "
        "--SpatialMatching.max_num_neighbors 30 "
        "--SpatialMatching.max_distance 100"
    )

    # open db, get image ids, write to frame ids, export frames to colmap
    model_path = os.path.join(output_path, "prior")
    has_pose_prior = False
    if has_pose_prior:
        frames_to_colmap(frames, model_path, output_format)
        os.makedirs(os.path.join(output_path, "sparse"), exist_ok=True)
        options = ""
        for i in range(cycles):
            os.system(
                f"colmap point_triangulator --database_path {output_path}/database.db "
                f"--input_path {model_path} --image_path {image_path} "
                f'--output_path {os.path.join(output_path, "sparse")}'
            )

            if i > 0:
                options = " --BundleAdjustment.refine_principal_point 1"
            os.system(
                f"colmap bundle_adjuster "
                f"--input_path {model_path} --image_path {image_path} "
                f'--output_path {os.path.join(output_path, "sparse")}'
                + options
            )
    else:
        # normal reconstruction with given intrinsic & shared cam param
        os.makedirs(os.path.join(output_path, "sparse"), exist_ok=True)
        options = " --Mapper.ba_refine_principal_point 1"
        sparse_path = os.path.join(output_path, "sparse")
        os.system(
            f"colmap mapper "
            f"--database_path {output_path}/database.db "
            f"--image_path {image_path} "
            f"--output_path {sparse_path} "
            f"--Mapper.abs_pose_min_inlier_ratio {0.35} "
            f"--Mapper.filter_max_reproj_error {3.0}"
        )

    # TODO dense model
    input_path = os.path.join(output_path, "sparse", "0")
    dense_path = os.path.join(output_path, "dense")

    os.makedirs(dense_path, exist_ok=True)
    os.system(
        f"colmap image_undistorter --image_path {image_path} "
        f"--input_path {input_path} "
        f"--output_path {dense_path}"
    )
    os.system(
        f"colmap patch_match_stereo --workspace_path {dense_path} "
        f"--PatchMatchStereo.geom_consistency true"
    )
    result_path = os.path.join(output_path, "fused.ply")
    os.system(
        f"colmap stereo_fusion --workspace_path {dense_path} "
        f"--output_path {result_path}"
    )
    print("end mapper reconstruction")

    return frames, None 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K SfM tool")
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
        help="Path to output reconstruction.",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default=None,
        help="Path to info file for sequence.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=".txt",
        choices=[".txt", ".bin"],
        help="Path to info file for sequence.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the SfM model after reconstruction.",
    )
    args = parser.parse_args()
    print(args.image_path)
    print(args.output_path)
    sfm_reconstruction(
        args.image_path, args.output_path, args.info_path, args.output_format
    )
    
