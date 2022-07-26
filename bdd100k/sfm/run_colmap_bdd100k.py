"""Scripts for running COLMAP on BDD100k data by postcodes."""
import argparse
import glob
import json
import os
import time
try:
    import cv2
except ImportError:
    pass
import numpy as np
from PIL import Image
from bdd100k.sfm.colmap.read_write_dense import read_array
from bdd100k.sfm.colmap.read_write_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)

from .run_colmap import (
    database_creator,
    dense_recon,
    feature_extractor,
    feature_matcher,
    image_deleter,
    new_mapper,
    orientation_aligner,
    stereo_fusion,
)
from .utils import frames_from_images, get_gps_priors


def get_sky_mask(seg_mask):
    """Sky mask"""
    sky_mask = np.ones(seg_mask.shape)
    sky_mask[seg_mask == 10] = 0
    return sky_mask


def get_info_path_from_images(info_base_path, image_tags_path):
    """Get the information (.json) file for each sequence."""
    info_path = []
    for img in os.listdir(image_tags_path):
        info_path.append(os.path.join(info_base_path, img[:-4] + ".json"))
    return info_path


def check_sparse_exist(sparse_path):
    """Check if sparse recon exists."""
    sparse_recon_exists = True
    orientation_aligned_path = os.path.join(sparse_path, "orientation_aligned")
    if os.path.exists(orientation_aligned_path):
        for path_ in os.listdir(orientation_aligned_path):
            if sparse_recon_exists and os.listdir(
                os.path.join(sparse_path, "orientation_aligned", path_)
            ):
                sparse_recon_exists = True
            elif not os.listdir(
                os.path.join(sparse_path, "orientation_aligned", path_)
            ):
                sparse_recon_exists = False
    else:
        sparse_recon_exists = False
    return sparse_recon_exists


def initialize_progress_tracker(target_path, save_path):
    """Create progress tracker."""
    singles_list = os.listdir(os.path.join(target_path, "singles"))
    overlaps_list = [
        int(i) for i in os.listdir(os.path.join(target_path, "overlaps"))
    ]
    overlaps_list.sort()
    singles_list.sort()

    progress_tracker = {
        "overlaps": {
            "Finished?": False,
            "Done": [],
            "Doing": [],
            "Failed": [],
            "Left": overlaps_list,
            "All": overlaps_list,
        },
        "singles": {
            "Finished?": False,
            "Done": [],
            "Doing": [],
            "Failed": [],
            "Left": singles_list,
            "All": singles_list,
        },
    }
    with open(save_path, "w", encoding="utf8") as fp:
        json.dump(progress_tracker, fp, indent=2)


def valiate_progress_tracker(target_path, progress_path, input_type):
    """
    If progress tracker is empty, initialize it and check which
    job is done or left
    """
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)
    base_dir = os.path.join(target_path, input_type)
    folder_list = os.listdir(base_dir)
    folder_list.sort()
    for folder in folder_list:
        database_path = os.path.join(base_dir, folder)
        if os.path.exists(
            os.path.join(database_path, "database.db")
        ) and not os.path.exists(
            os.path.join(database_path, "database.db-shm")
        ):
            progress[input_type]["Left"].remove(int(folder))
            progress[input_type]["Done"].append(int(folder))

    with open(progress_path, "w", encoding="utf8") as fp:
        json.dump(progress, fp, indent=2)


def progress_get_next(progress_path, input_type, check_path=None):
    """Get the next target to proceed in progress tracker."""
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)
    next_element = progress[input_type]["Left"].pop(0)
    if check_path is not None:
        with open(check_path, "r", encoding="utf8") as fp:
            check = json.load(fp)
        while next_element not in check[input_type]["Done"]:
            progress[input_type]["Left"].append(next_element)
            next_element = progress[input_type]["Left"].pop(0)

    progress[input_type]["Doing"].append(next_element)
    with open(progress_path, "w", encoding="utf8") as fp:
        json.dump(progress, fp, indent=2)
    return next_element


def progress_update(progress_path, input_type, element):
    """Update the finished target in the progress tracker."""
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)
    try:
        progress[input_type]["Doing"].remove(element)
    except KeyError:
        print(f"This element {element} is not in Doing.")
        return
    progress[input_type]["Done"].append(element)
    if (
        len(progress[input_type]["Left"]) == 0
        and len(progress[input_type]["Doing"]) == 0
    ):
        progress[input_type]["Finished?"] = True
    with open(progress_path, "w", encoding="utf8") as fp:
        json.dump(progress, fp, indent=2)


def failure_update(progress_path, input_type, element):
    """Update the failed target in the progress tracker."""
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)
    try:
        progress[input_type]["Doing"].remove(element)
    except KeyError:
        print(f"This element {element} is not in Doing")
        return
    progress[input_type]["Failed"].append(element)
    if (
        len(progress[input_type]["Left"]) == 0
        and len(progress[input_type]["Doing"]) == 0
    ):
        progress[input_type]["Finished?"] = True
    with open(progress_path, "w", encoding="utf8") as fp:
        json.dump(progress, fp, indent=2)


def is_progress_finished(progress_path, input_type):
    """Check if we are done in the progress tracker."""
    # If the progress tracker is empty, initialize it
    if os.stat(progress_path).st_size == 0:
        job = progress_path.split("/")[-1].split("_")[0]
        if job == "feature":
            initialize_progress_tracker(
                os.path.dirname(progress_path), progress_path
            )
            valiate_progress_tracker(
                os.path.dirname(progress_path), progress_path, input_type
            )
    with open(progress_path, "r", encoding="utf8") as fp:
        progress = json.load(fp)

    return (
        progress[input_type]["Finished?"]
        or len(progress[input_type]["Left"]) == 0
    )


def write_list_to_txt(input_list, output_path):
    """Used for delete frames from sparse reconstruction."""
    with open(output_path, "w", encoding="utf8") as f:
        for ele in input_list:
            f.write(str(ele))
            f.write("\n")


def get_fusion_mask(depth_map, seg_mask):
    """Update the image masks for stereo fusion."""
    mask_fusion = np.ones(depth_map.shape, dtype=np.int8)
    # Percentile filter
    min_depth, max_depth = np.percentile(depth_map, [5, 95])
    mask_fusion[depth_map < min_depth] = 0
    mask_fusion[depth_map > max_depth] = 0
    # Only keep of points within 3 - 250 meter
    mask_fusion[depth_map < 3] = 0
    mask_fusion[depth_map > 250] = 0

    car_interier_corp = np.zeros(depth_map.shape)
    # Remove lower part of images for car interior
    car_interier_corp[520:][:] = 1.0
    car_interier_noise_mask = np.zeros(depth_map.shape)
    car_interier_noise_mask[520:][:] = 1.0
    car_interier_noise_mask[depth_map > 15] = 0.0
    car_interier_noise_mask = car_interier_noise_mask + (1 - car_interier_corp)

    sky_mask = get_sky_mask(seg_mask)
    mask_fusion = car_interier_noise_mask * mask_fusion * sky_mask
    return mask_fusion


def create_fusion_mask(dense_path, seg_mask_path):
    """Create a image masks only for stereo fusion"""
    print(f"Creating fusion mask for: {dense_path}")
    depth_path = os.path.join(dense_path, "stereo", "depth_maps")
    depth_maps_geometric = glob.glob(depth_path + "/*.geometric.bin")
    fusion_mask_path = os.path.join(dense_path, "fusion_mask")
    os.makedirs(fusion_mask_path, exist_ok=True)
    for depth_map_path in depth_maps_geometric:
        image_name = ".".join(depth_map_path.split("/")[-1].split(".")[:2])

        if not os.path.exists(depth_map_path):
            print(f"File not found: {depth_map_path}")
            continue
        depth_map = read_array(depth_map_path)
        # Get segmentation mask
        seg_mask_name = image_name[:-4] + ".png"
        seg_mask_file = os.path.join(seg_mask_path, seg_mask_name)
        seg_mask = np.array(Image.open(seg_mask_file))
        # Create fusion mask
        mask_fusion = get_fusion_mask(depth_map, seg_mask)
        cv2.imwrite(f"{fusion_mask_path}/{image_name}.png", mask_fusion)
    return fusion_mask_path


def check_sparse_recon(path, ext=".bin"):
    """Post process step to chech if sparse reconstruction is successful"""
    sparse_aligned_folder = os.path.dirname(path)
    num_folder = os.path.basename(path)
    sparse_folder = os.path.dirname(sparse_aligned_folder)
    sparse_original_folder = os.path.join(sparse_folder, num_folder)

    cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
    images = read_images_binary(os.path.join(path, "images" + ext))
    # Check if there are enough frames
    length_check = True
    if len(images) < 50:
        length_check = False
        return length_check, ""

    # Check if focal length is correctly estimated
    focal_length_check = True
    if len(cameras) == 1 and (1 in cameras):
        focal_length = cameras[1].params[0]
        if not 950.0 <= focal_length <= 1100.0:
            focal_length_check = False
            return focal_length_check, ""
    else:
        focal_length_check = False
        return focal_length_check, ""

    # Check if there is discontinuity
    # If there is a discontinuity, then the whole squence will be removed
    continuity_check = True
    failed_sequence = []
    failed_frame_names = []
    failed_frame_id = []
    key_last_valid = sorted(images)[0]
    for key in sorted(images):
        if key == key_last_valid:
            continue

        if images[key].name[:17] != images[key_last_valid].name[:17]:
            key_last_valid = key
            continue
        # calculate distance traveled from the translation vector
        cur_tvec = images[key].tvec
        cur_r = qvec2rotmat(images[key].qvec)
        last_tvec = images[key_last_valid].tvec
        last_r = qvec2rotmat(images[key_last_valid].qvec)
        cur_t = -cur_r.T @ cur_tvec
        last_t = -last_r.T @ last_tvec
        dist_moved = np.linalg.norm(cur_t - last_t)

        if dist_moved > 12:
            failed_sequence.append(images[key].name[:17])
            failed_frame_names.append(images[key].name)
            failed_frame_id.append(images[key].id)
        else:
            key_last_valid = key

    num_images = len(images)
    num_failed_images = len(failed_frame_names)
    sucess_percentage_images = 1 - num_failed_images / num_images
    # Arbitrarily decide that if 52% of frames are failed, discard recon
    continuity_check = sucess_percentage_images > 0.48
    check_result = length_check and focal_length_check and continuity_check

    if check_result:
        success_sparse_folder = os.path.join(
            sparse_folder, "successful", num_folder
        )
        os.makedirs(success_sparse_folder, exist_ok=True)
        init_file = os.path.join(sparse_original_folder, "project.ini")
        os.system(f"cp {init_file} " f"{success_sparse_folder}")
        if sucess_percentage_images == 1.0:
            sparese_files = os.path.join(path, "*")
            os.system(f"cp {sparese_files} " f"{success_sparse_folder}")
        else:
            write_list_to_txt(
                failed_frame_names,
                os.path.join(path, "failed_image_names.txt"),
            )
            write_list_to_txt(
                failed_frame_id, os.path.join(path, "failed_image_id.txt")
            )
            image_deleter(
                path,
                success_sparse_folder,
                os.path.join(path, "failed_image_id.txt"),
            )
        return check_result, success_sparse_folder
    return check_result, ""


def check_dense_exist(dense_path):
    """Check if dense folder exists and consists of all outputs"""
    dense_exists = False
    if os.path.isdir(dense_path):
        for path_ in os.listdir(dense_path):
            if (
                os.path.exists(os.path.join(dense_path, path_, "fused.ply"))
                and os.path.exists(os.path.join(dense_path, path_, "images"))
                and os.path.exists(os.path.join(dense_path, path_, "sparse"))
                and os.path.exists(os.path.join(dense_path, path_, "stereo"))
            ):
                dense_exists = True
                normal_path = os.path.join(
                    dense_path, path_, "stereo", "normal_maps"
                )
                if os.path.exists(normal_path):
                    remove_path(normal_path)
    return dense_exists


def remove_path(target_path):
    """Delete a path"""
    print(f"Start to remove {target_path}")
    os.system(f"rm -rf {target_path}")


def move_path(target_path, destination_path):
    """Migrate a path"""
    print(f"Start to move {target_path} to {destination_path}")
    os.system(f"mv {target_path} {destination_path}")


def parse_args():
    """All Arguments"""
    parser = argparse.ArgumentParser(
        description="Conduct colmap reconstruction on BDD100K data"
    )
    parser.add_argument(
        "--postcode",
        type=str,
        help="Which postcode to run. e.g. 10009",
    )
    parser.add_argument(
        "--timeofday",
        type=str,
        default="daytime",
        help="Which timeofday to process(daytime, night, dawn_dusk)",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="overlaps",
        help="Enter overlaps or singles. One job process either one of two",
    )
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        default="feature",
        help="Which job to do",
    )
    parser.add_argument(
        "--matcher-method",
        type=str,
        default="spatial",
        help="The feature match method. (spatial, sequential, exhaustive).",
    )
    parser.add_argument(
        "--target-path",
        "-t",
        type=str,
        default="/scratch_net/zoidberg_second/yuthan/bdd100k/",
        help="Local path to save all postcodes and the results.",
    )
    parser.add_argument(
        "--info-base-path",
        type=str,
        default="/srv/beegfs02/scratch/bdd100k/data/bdd100k/info",
        help="Path to all info .json for bdd100k.",
    )
    parser.add_argument(
        "--colmap-path",
        type=str,
        default="colmap",
        help="The path to the modeified colmap.",
    )
    parser.add_argument(
        "--destination-path",
        type=str,
        default="/srv/beegfs02/scratch/bdd100k/data/sfm/postcode",
        help="The path to the server where all results are stored.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Run colmap with out GPU (Only works for sparse reconstruction)",
    )
    args = parser.parse_args()
    return args


def main():
    """Conduct colmap reconstruction on BDD100K data"""
    args = parse_args()

    colmap_new = args.colmap_path
    target_path = args.target_path
    info_base_path = args.info_base_path
    matcher_method = args.matcher_method
    postcode_path = os.path.join(target_path, args.postcode)
    timeofday_path = os.path.join(postcode_path, args.timeofday)
    input_type = args.input_type
    destination_path = os.path.join(
        args.destination_path, args.postcode, args.timeofday, input_type
    )

    # type path is either overlaps or singles path
    type_path = os.path.join(timeofday_path, input_type)

    feature_progress = os.path.join(timeofday_path, "feature_progress.json")
    sparse_progress = os.path.join(timeofday_path, "sparse_progress.json")
    dense_progress = os.path.join(timeofday_path, "dense_progress.json")

    if not os.path.exists(feature_progress):
        initialize_progress_tracker(timeofday_path, feature_progress)
    if not os.path.exists(sparse_progress):
        initialize_progress_tracker(timeofday_path, sparse_progress)
    if not os.path.exists(dense_progress):
        initialize_progress_tracker(timeofday_path, dense_progress)
    if args.job == "feature":
        while not is_progress_finished(feature_progress, input_type):
            i = progress_get_next(feature_progress, input_type)
            seq_path = os.path.join(type_path, str(i))
            output_path = seq_path
            image_path = os.path.join(seq_path, "images")
            image_tags_path = os.path.join(seq_path, "image_tags")
            info_path = get_info_path_from_images(
                info_base_path, image_tags_path
            )

            try:
                frames = get_gps_priors(info_path, image_path)
            except KeyError:
                print("Key Error: Some images data is missing")
            if frames is None:
                frames = frames_from_images(image_path)
            database_creator(output_path, colmap_new)
            while not os.path.exists(f"{output_path}/database.db"):
                time.sleep(0.5)
            frames = feature_extractor(
                frames,
                image_path,
                output_path,
                colmap_new,
                args.no_gpu,
            )
            # Used for spatial matcher
            max_num_neighbors = min(160, int(len(os.listdir(image_path)) / 4))
            frames = feature_matcher(
                frames,
                matcher_method,
                output_path,
                colmap_new,
                max_num_neighbors,
                args.no_gpu,
            )
            print("Complete feature extractor and matcher for:  " + str(i))
            progress_update(feature_progress, input_type, i)

    elif args.job == "mapper":
        while not is_progress_finished(sparse_progress, input_type):
            i = progress_get_next(
                sparse_progress, input_type, feature_progress
            )
            seq_path = os.path.join(type_path, str(i))
            output_path = seq_path
            image_path = os.path.join(seq_path, "images")
            sparse_path = os.path.join(output_path, "sparse")
            os.makedirs(sparse_path, exist_ok=True)

            new_mapper(image_path, output_path, sparse_path, colmap_new)
            print("Complete sparse reconstruction for:  " + str(i))

            sparse_results = os.listdir(sparse_path)
            for folder in sparse_results:
                orientation_aligned_path = os.path.join(
                    sparse_path,
                    "orientation_aligned",
                    f"{folder}",
                )
                os.makedirs(orientation_aligned_path, exist_ok=True)
                orientation_aligner(
                    image_path,
                    os.path.join(sparse_path, folder),
                    orientation_aligned_path,
                    colmap_new,
                )
            if check_sparse_exist(sparse_path):
                progress_update(sparse_progress, input_type, i)
            else:
                failure_update(sparse_progress, input_type, i)

    elif args.job == "dense":
        os.makedirs(destination_path, exist_ok=True)
        while not is_progress_finished(dense_progress, input_type):
            i = progress_get_next(dense_progress, input_type)
            seq_path = os.path.join(type_path, str(i))
            output_path = seq_path
            image_path = os.path.join(seq_path, "images")
            seg_mask_path = os.path.join(seq_path, "seg_mask")
            # use aligned sparse output as input for dense model
            sparse_aligned_path = os.path.join(
                output_path, "sparse", "orientation_aligned"
            )
            sparse_results = os.listdir(sparse_aligned_path)
            # folder here is named after an int or seq name
            for folder in sparse_results:
                input_path = os.path.join(sparse_aligned_path, folder)
                # Make sure sparse exists
                if os.listdir(input_path):
                    # Check if sparse reconstruction is correct
                    (
                        sparse_is_successful,
                        successful_sparse_path,
                    ) = check_sparse_recon(input_path)
                    if not sparse_is_successful:
                        continue

                    # Dense reconstruction starts
                    dense_path = os.path.join(
                        output_path, "dense", f"{folder}_dense"
                    )
                    os.makedirs(dense_path, exist_ok=True)
                    dense_recon(
                        image_path,
                        successful_sparse_path,
                        dense_path,
                        colmap_new,
                    )

                    # Stereo fusion
                    fusion_mask_path = create_fusion_mask(
                        dense_path, seg_mask_path
                    )
                    result_path = os.path.join(dense_path, "fused.ply")
                    stereo_fusion(
                        dense_path, result_path, fusion_mask_path, colmap_new
                    )
                else:
                    print(f"Aborted: Sparse recon {folder} does not exist")
                    continue

            if check_dense_exist(os.path.join(output_path, "dense")):
                # Migrate results to the server
                move_path(output_path, destination_path)
                remove_path(seq_path)
                progress_update(dense_progress, input_type, i)
            else:
                failure_update(dense_progress, input_type, i)


if __name__ == "__main__":
    main()
