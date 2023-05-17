"""Scripts for running COLMAP on BDD100k or any sequences."""
import argparse
import os
import time
from typing import List, Optional

from scalabel.label.typing import Frame
from scalabel.label.utils import get_matrix_from_extrinsics

from .colmap.database_io import COLMAPDatabase  # type: ignore
from .utils import (
    cam_spec_prior,
    create_fusion_masks_pan,
    frames_from_images,
    get_gps_priors,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run COLMAP Reconstruction for image sequences"
    )
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        default="feature",
        help="Which job to do(feature, mapper, stereo_fusion, all, etc)",
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
    parser.add_argument(
        "--intrinsics",
        type=str,
        default="bdd100k",
        help="Path to a json file for camera intriniscs.",
    )
    parser.add_argument(
        "--no-prior-motion",
        action="store_true",
        help="To not use prior knowledge on GPS in Bundle adjustment",
    )
    ARGUMENTS = parser.parse_args()


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
    """Add the spatial locations to images in database.

    (already transformed to cartesian)
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
    intrinsics_path: Optional[str] = "bdd100k",
    mask_path: Optional[str] = "",
) -> List[Frame]:
    """Conduct feature extraction."""
    # we assume shared intrinsics
    intrinsics = None
    if frames[0].intrinsics is not None:
        assert frames[0].intrinsics.focal[0] == frames[0].intrinsics.focal[0]
        intrinsics = frames[0].intrinsics
    else:
        intrinsics = cam_spec_prior(intrinsics_path)

    no_gpu_option = " --SiftExtraction.use_gpu 0" if no_gpu else ""
    mask_option = (
        f" --ImageReader.mask_path {mask_path}" if mask_path != "" else ""
    )
    if intrinsics:
        f_x = intrinsics.focal[0]
        c_x, c_y = intrinsics.center
        intrinsics_options = (
            " --ImageReader.camera_model SIMPLE_PINHOLE"
            " --ImageReader.single_camera 1"
            f" --ImageReader.camera_params {f_x},{c_y},{c_x}"
        )
    else:
        intrinsics_options = " --ImageReader.camera_model SIMPLE_PINHOLE"
    os.system(
        f"{colmap_new} feature_extractor "
        f"--database_path {output_path}/database.db "
        f"--image_path {image_path}"
        + intrinsics_options
        + no_gpu_option
        + mask_option
    )
    return frames


def database_creator(
    output_path: str,
    colmap_new: str,
) -> None:
    """Create colmap database."""
    database_path = os.path.join(output_path, "database.db")
    os.system(
        f"{colmap_new} database_creator" f" --database_path {database_path}"
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


def run_feature() -> List[Frame]:
    """Conduct feature extractor and feature matcher."""
    if ARGUMENTS.info_path is None:
        frames = frames_from_images(ARGUMENTS.image_path)
    else:
        frames = get_gps_priors(
            ARGUMENTS.info_path, ARGUMENTS.image_path, ARGUMENTS.intrinsics
        )
    database_creator(ARGUMENTS.output_path, ARGUMENTS.colmap_path)
    while not os.path.exists(f"{ARGUMENTS.output_path}/database.db"):
        time.sleep(0.5)
    frames = feature_extractor(
        frames,
        ARGUMENTS.image_path,
        ARGUMENTS.output_path,
        ARGUMENTS.colmap_path,
        ARGUMENTS.no_gpu,
        ARGUMENTS.intrinsics,
        ARGUMENTS.mask_path,
    )
    # Used for spatial matcher, only match 160 nearest frames at most
    max_num_neighbors = min(
        160, int(len(os.listdir(ARGUMENTS.image_path)) / 4)
    )
    frames = feature_matcher(
        frames,
        ARGUMENTS.matcher_method,
        ARGUMENTS.output_path,
        ARGUMENTS.colmap_path,
        max_num_neighbors,
        ARGUMENTS.no_gpu,
    )
    return frames


def new_mapper(
    image_path: str,
    output_path: str,
    sparse_path: str,
    colmap_new: str,
    no_prior_motion: Optional[bool] = False,
) -> None:
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
        f"--Mapper.use_prior_motion {int(not no_prior_motion)} "
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


def image_deleter(
    input_path: str,
    output_path: str,
    image2delete_id_path: str,
) -> None:
    """Delete images from sparse reconstruction."""
    os.system(
        f"colmap image_deleter "
        f"--input_path {input_path} "
        f"--output_path {output_path} "
        f"--image_ids_path {image2delete_id_path}"
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
    """Conterduct seo fusion."""
    if len(mask_path) != 0:
        options = f"--StereoFusion.mask_path {mask_path} "
    else:
        options = ""

    os.system(
        f"{colmap_new} stereo_fusion --workspace_path {dense_path} "
        f"--output_path {result_path} " + options
    )


def main() -> None:
    """Run sparse reconstruction."""
    sparse_path = os.path.join(ARGUMENTS.output_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)

    if ARGUMENTS.job == "feature":
        run_feature()
    elif ARGUMENTS.job == "mapper":
        new_mapper(
            ARGUMENTS.image_path,
            ARGUMENTS.output_path,
            sparse_path,
            ARGUMENTS.colmap_path,
            ARGUMENTS.no_prior_motion,
        )
    elif ARGUMENTS.job == "orien_aligner":
        sparse_results = os.listdir(sparse_path)
        for folder in sparse_results:
            orientation_aligned_path = os.path.join(
                sparse_path,
                "orientation_aligned",
                f"{folder}",
            )
            os.makedirs(orientation_aligned_path, exist_ok=True)
            orientation_aligner(
                ARGUMENTS.image_path,
                os.path.join(sparse_path, folder),
                orientation_aligned_path,
                ARGUMENTS.colmap_path,
            )
    elif ARGUMENTS.job == "sparse":
        run_feature()
        new_mapper(
            ARGUMENTS.image_path,
            ARGUMENTS.output_path,
            sparse_path,
            ARGUMENTS.colmap_path,
            ARGUMENTS.no_prior_motion,
        )
        sparse_results = os.listdir(sparse_path)
        for folder in sparse_results:
            orientation_aligned_path = os.path.join(
                sparse_path,
                "orientation_aligned",
                f"{folder}",
            )
            os.makedirs(orientation_aligned_path, exist_ok=True)
            orientation_aligner(
                ARGUMENTS.image_path,
                os.path.join(sparse_path, folder),
                orientation_aligned_path,
                ARGUMENTS.colmap_path,
            )
    elif ARGUMENTS.job == "dense":
        sparse_aligned_path = os.path.join(sparse_path, "orientation_aligned")
        sparse_results = os.listdir(sparse_aligned_path)
        for folder in sparse_results:
            input_path = os.path.join(sparse_aligned_path, folder)
            dense_path = os.path.join(
                ARGUMENTS.output_path, "dense", f"{folder}_dense"
            )
            os.makedirs(dense_path, exist_ok=True)
            dense_recon(
                ARGUMENTS.image_path,
                input_path,
                dense_path,
                ARGUMENTS.colmap_path,
            )
    elif ARGUMENTS.job == "stereo_fusion":
        sparse_aligned_path = os.path.join(sparse_path, "orientation_aligned")
        sparse_results = os.listdir(sparse_aligned_path)
        pan_mask_path = os.path.join(ARGUMENTS.output_path, "pan_mask")
        for folder in sparse_results:
            dense_path = os.path.join(
                ARGUMENTS.output_path, "dense", f"{folder}_dense"
            )
            if ARGUMENTS.mask_path == "":
                fusion_mask_path = create_fusion_masks_pan(
                    dense_path, pan_mask_path
                )
            else:
                fusion_mask_path = ARGUMENTS.mask_path
            result_path = os.path.join(dense_path, "fused.ply")
            stereo_fusion(
                dense_path,
                result_path,
                fusion_mask_path,
                ARGUMENTS.colmap_path,
            )
    elif ARGUMENTS.job == "all":
        run_feature()
        new_mapper(
            ARGUMENTS.image_path,
            ARGUMENTS.output_path,
            sparse_path,
            ARGUMENTS.colmap_path,
            ARGUMENTS.no_prior_motion,
        )
        sparse_results = os.listdir(sparse_path)
        for folder in sparse_results:
            orientation_aligned_path = os.path.join(
                sparse_path,
                "orientation_aligned",
                f"{folder}",
            )
            os.makedirs(orientation_aligned_path, exist_ok=True)
            orientation_aligner(
                ARGUMENTS.image_path,
                os.path.join(sparse_path, folder),
                orientation_aligned_path,
                ARGUMENTS.colmap_path,
            )
        sparse_aligned_path = os.path.join(sparse_path, "orientation_aligned")
        sparse_results = os.listdir(sparse_aligned_path)
        pan_mask_path = os.path.join(ARGUMENTS.output_path, "pan_mask")
        for folder in sparse_results:
            input_path = os.path.join(sparse_aligned_path, folder)
            dense_path = os.path.join(
                ARGUMENTS.output_path, "dense", f"{folder}_dense"
            )
            os.makedirs(dense_path, exist_ok=True)
            dense_recon(
                ARGUMENTS.image_path,
                input_path,
                dense_path,
                ARGUMENTS.colmap_path,
            )
            if ARGUMENTS.mask_path == "":
                fusion_mask_path = create_fusion_masks_pan(
                    dense_path, pan_mask_path
                )
            else:
                fusion_mask_path = ARGUMENTS.mask_path
            result_path = os.path.join(dense_path, "fused.ply")
            stereo_fusion(
                dense_path,
                result_path,
                fusion_mask_path,
                ARGUMENTS.colmap_path,
            )


if __name__ == "__main__":
    main()
