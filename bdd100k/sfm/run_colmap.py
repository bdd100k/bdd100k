"""Scripts for running COLMAP on BDD100k or any sequences."""
import argparse
import os
import time
from typing import List, Optional

from scalabel.label.typing import Frame
from scalabel.label.utils import get_matrix_from_extrinsics

from .colmap.database_io import COLMAPDatabase
from .utils import cam_spec_prior, frames_from_images, get_gps_priors


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
    """
    Add the spatial locations (transformed to cartesian) to images in database.
    """
    db = COLMAPDatabase.connect(db_path)
    for f in frames:
        if f.extrinsics is not None and f.attributes is not None:
            trans_mat = get_matrix_from_extrinsics(f.extrinsics)
            tvec = trans_mat[:3, 3]
            [tx, ty, tz] = tvec
            image_id = f.attributes["id"]
            db.execute(
                f"""UPDATE images
                SET prior_tx={tx}, prior_ty={ty}, prior_tz={tz}
                Where image_id={image_id}"""
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
            image_id = f.attributes["id"]
            db.execute(
                f"""UPDATE images
                SET prior_tx={tx}, prior_ty={ty}, prior_tz={tz}
                Where image_id={image_id}"""
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
    colmap_new: str,
    no_gpu: Optional[bool] = False,
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

    no_gpu_option = (" --SiftExtraction.use_gpu 0") if no_gpu else ("")

    options = (
        " --ImageReader.camera_model SIMPLE_PINHOLE"
        " --ImageReader.single_camera 1"
        f" --ImageReader.camera_params {f_x},{c_y},{c_x}"
    )
    os.system(
        f"{colmap_new} feature_extractor "
        f"--database_path {output_path}/database.db "
        f"--image_path {image_path}" + options + no_gpu_option
    )
    return frames


def database_creator(
    output_path: str,
    colmap_new: str,
) -> None:
    """Create colmap database."""
    os.system(
        f"{colmap_new} database_creator"
        f" --database_path {output_path}/database.db"
    )


def feature_matcher(
    frames: List[Frame],
    matcher_method: str,
    output_path: str,
    colmap_new: str,
    num_neighbors: Optional[int] = 150,
    no_gpu: Optional[bool] = False,
) -> List[Frame]:
    """Conduct feature matcher."""
    add_image_ids(frames, f"{output_path}/database.db")
    add_gps_to_db(frames, f"{output_path}/database.db")
    no_gpu_option = (" --SiftExtraction.use_gpu 0") if no_gpu else ("")
    if matcher_method == "exhaustive":
        os.system(
            f"{colmap_new} exhaustive_matcher "
            f"--database_path {output_path}/database.db "
            f"--SiftMatching.min_inlier_ratio {0.65} "
            f"--SiftMatching.max_distance {0.55} "
            f"--SiftMatching.min_num_inliers {50}" + no_gpu_option
        )
    elif matcher_method == "spatial":
        os.system(
            f"{colmap_new} spatial_matcher "
            f"--database_path {output_path}/database.db "
            f"--SiftMatching.min_inlier_ratio {0.25} "
            f"--SiftMatching.min_num_inliers {15} "
            "--SpatialMatching.is_gps 1 "
            f"--SpatialMatching.max_num_neighbors {num_neighbors} "
            "--SpatialMatching.max_distance 100" + no_gpu_option
        )
    elif matcher_method == "sequential":
        os.system(
            f"{colmap_new} sequential_matcher "
            f"--database_path {output_path}/database.db "
            f"--SiftMatching.min_inlier_ratio {0.65} "
            f"--SiftMatching.max_distance {0.55} "
            f"--SiftMatching.min_num_inliers {60}" + no_gpu_option
        )
    return frames


def new_mapper(
    image_path: str,
    output_path: str,
    sparse_path: str,
    colmap_new: str,
):
    """Conduct incremental mapper using the modified colmap with GPS."""
    os.system(
        f"{colmap_new} mapper "
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
        " --Mapper.ba_refine_principal_point 1"
    )


def orientation_aligner(
    image_path: str,
    sparse_path: str,
    output_path: str,
    colmap_new: str,
) -> None:
    """Conduct orientation aligner."""
    os.system(
        f"{colmap_new} model_orientation_aligner "
        f"--image_path {image_path} "
        f"--input_path {sparse_path} "
        f"--output_path {output_path} "
    )


def dense_recon(
    image_path: str,
    input_path: str,
    dense_path: str,
    colmap_new: str,
) -> None:
    """Conduct dense reconstruction."""
    os.system(
        f"{colmap_new} image_undistorter --image_path {image_path} "
        f"--input_path {input_path} "
        f"--output_path {dense_path}"
    )
    os.system(
        f"{colmap_new} patch_match_stereo --workspace_path {dense_path} "
        f"--PatchMatchStereo.geom_consistency true"
    )


def stereo_fusion(
    dense_path: str,
    result_path: str,
    mask_path: str,
    colmap_new: str,
) -> None:
    """Conduct stereo fusion."""
    if len(mask_path) != 0:
        options = f"--StereoFusion.mask_path {mask_path} "
    else:
        options = ""

    os.system(
        f"{colmap_new} stereo_fusion --workspace_path {dense_path} "
        f"--output_path {result_path} " + options
    )


def main():
    """Run sparse reconstruction."""
    parser = argparse.ArgumentParser(
        description="Sparse Reconstruction for a sequence"
    )
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        default="feature",
        help="Which job to do(feature, mapper, orien_aligner or all)",
    )
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
        action="append",
        default=None,
        help="""
        Path to info file in .json for sequence.
        """,
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default="",
        help="""
        Path to image mask for stereo fusion.
        """,
    )
    parser.add_argument(
        "--matcher-method",
        type=str,
        default="spatial",
        help="The feature match method. (spatial, sequential, exhaustive).",
    )
    parser.add_argument(
        "--colmap-path",
        default="colmap",
        type=str,
        help="The path to the modeified colmap.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Run colmap with out GPU (Only works for sparse reconstruction)",
    )

    args = parser.parse_args()
    colmap_new = args.colmap_path
    sparse_path = os.path.join(args.output_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    orien_aligned_path = os.path.join(sparse_path, "orientation_aligned")
    os.makedirs(orien_aligned_path, exist_ok=True)

    
    
    if args.job == "feature":
        frames = get_gps_priors(args.info_path, args.image_path)
        if args.info_path is None:
            frames = frames_from_images(args.image_path)
        database_creator(args.output_path, colmap_new)
        while not os.path.exists(f"{args.output_path}/database.db"):
            time.sleep(0.5)
        frames = feature_extractor(
            frames,
            args.image_path,
            args.output_path,
            colmap_new,
            args.no_gpu,
        )
        frames = feature_matcher(
            frames,
            args.matcher_method,
            args.output_path,
            colmap_new,
            50,
            args.no_gpu,
        )
    elif args.job == "mapper":
        new_mapper(args.image_path, args.output_path, sparse_path, colmap_new)
        orientation_aligner(
            args.image_path,
            os.path.join(sparse_path, "0"),
            orien_aligned_path,
            colmap_new,
        )
    elif args.job == "orien_aligner":
        orientation_aligner(
            args.image_path,
            os.path.join(sparse_path, "0"),
            orien_aligned_path,
            colmap_new,
        )
    elif args.job == "sparse":
        frames = get_gps_priors(args.info_path, args.image_path)
        if args.info_path is None:
            frames = frames_from_images(args.image_path)
        database_creator(args.output_path, colmap_new)
        while not os.path.exists(f"{args.output_path}/database.db"):
            time.sleep(0.5)
        frames = feature_extractor(
            frames,
            args.image_path,
            args.output_path,
            colmap_new,
            args.no_gpu,
        )
        frames = feature_matcher(
            frames,
            args.matcher_method,
            args.output_path,
            colmap_new,
            50,
            args.no_gpu,
        )
        new_mapper(args.image_path, args.output_path, sparse_path, colmap_new)
        orientation_aligner(
            args.image_path,
            os.path.join(sparse_path, "0"),
            orien_aligned_path,
            colmap_new,
        )
    elif args.job == "dense":
        dense_path = os.path.join(args.output_path, "dense")
        os.makedirs(dense_path, exist_ok=True)
        dense_recon(
            args.image_path,
            orien_aligned_path,
            dense_path,
            colmap_new,
        )
    elif args.job == "stereo_fusion":
        dense_path = os.path.join(args.output_path, "dense")
        result_path = os.path.join(dense_path, "fused.ply")
        stereo_fusion(dense_path, result_path, args.mask_path, colmap_new)
    elif args.job == "all":
        frames = get_gps_priors(args.info_path, args.image_path)
        if args.info_path is None:
            frames = frames_from_images(args.image_path)
        database_creator(args.output_path, colmap_new)
        while not os.path.exists(f"{args.output_path}/database.db"):
            time.sleep(0.5)
        frames = feature_extractor(
            frames,
            args.image_path,
            args.output_path,
            colmap_new,
            args.no_gpu,
        )
        frames = feature_matcher(
            frames,
            args.matcher_method,
            args.output_path,
            colmap_new,
            50,
            args.no_gpu,
        )
        new_mapper(args.image_path, args.output_path, sparse_path, colmap_new)
        orientation_aligner(
            args.image_path,
            os.path.join(sparse_path, "0"),
            orien_aligned_path,
            colmap_new,
        )
        dense_path = os.path.join(args.output_path, "dense")
        os.makedirs(dense_path, exist_ok=True)
        dense_recon(
            args.image_path,
            orien_aligned_path,
            dense_path,
            colmap_new,
        )
        result_path = os.path.join(dense_path, "fused.ply")
        stereo_fusion(dense_path, result_path, args.mask_path, colmap_new)


if __name__ == "__main__":
    main()
