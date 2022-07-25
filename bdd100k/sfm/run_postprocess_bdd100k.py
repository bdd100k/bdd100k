"""Post process the depth maps for bdd100k data by postcode"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import pylab as plt
from matplotlib import cm
from PIL import Image

from bdd100k.sfm.colmap.database_io import COLMAPDatabase
from bdd100k.sfm.colmap.read_write_dense import read_array, write_array
from bdd100k.sfm.colmap.read_write_model import qvec2rotmat, read_model

from .utils import gps_to_m


def plot_depth(depth_map, image_path="", title="", visualize=True):
    """Plot filtered depth"""
    depth_map_visual = np.ma.masked_where(depth_map == 0, depth_map)
    cmap = cm.Blues_r
    cmap.set_bad(color="gray")
    plt.figure(figsize=(30, 20))
    plt.imshow(depth_map_visual, cmap=cmap)
    plt.colorbar()
    plt.title(title)

    if visualize:
        plt.show()
    else:
        plt.imsave(image_path, depth_map_visual, cmap=cmap)


def plot_mask(mask, title=""):
    """Plot image mask"""
    mask_visual = np.ma.masked_where(mask == 0.0, mask)
    cmap = cm.Blues_r
    cmap.set_bad(color="gray")
    plt.figure(figsize=(30, 20))
    plt.imshow(mask_visual, cmap=cmap)
    plt.title(title)
    plt.show()


def plot_depth_original(depth_map):
    """Plot original depth"""
    plt.figure()
    plt.imshow(depth_map)
    plt.title("depth map")


def apply_range_filter(depth_map):
    """Apply range filtering"""
    depth_map[depth_map < 3] = 0.0
    depth_map[depth_map > 80] = 0.0

    car_interier_corp = np.zeros(depth_map.shape)
    car_interier_corp[520:][:] = 1.0

    car_interier_noise_mask = np.zeros(depth_map.shape)
    car_interier_noise_mask[520:][:] = 1.0
    car_interier_noise_mask[depth_map > 15] = 0.0
    car_interier_noise_mask = car_interier_noise_mask + (1 - car_interier_corp)
    depth_map = depth_map * car_interier_noise_mask
    return depth_map


def apply_clean_filter(depth_map, sky_mask):
    """Apply a range filter that has less restriction"""
    depth_map[depth_map < 3] = 0.0
    depth_map[depth_map > 200] = 0.0

    car_interier_corp = np.zeros(depth_map.shape)
    car_interier_corp[520:][:] = 1.0

    car_interier_noise_mask = np.zeros(depth_map.shape)
    car_interier_noise_mask[520:][:] = 1.0
    car_interier_noise_mask[depth_map > 15] = 0.0
    car_interier_noise_mask = car_interier_noise_mask + (1 - car_interier_corp)
    depth_map = depth_map * car_interier_noise_mask * sky_mask
    return depth_map


def apply_hsv_filter(depth_map, raw_image):
    """Apply color filter: filter out pixels with specific color"""
    imagergb = cv2.imread(raw_image)
    imagehsv = cv2.cvtColor(imagergb, cv2.COLOR_BGR2HSV)

    lower_sky = np.array([86, 13, 102])
    upper_sky = np.array([120, 180, 255])
    imagemask_sky = cv2.inRange(imagehsv, lower_sky, upper_sky)

    lower_dusk_sky = np.array([70, 28, 102])
    upper_dusk_sky = np.array([80, 50, 255])
    imagemask_dusk_sky = cv2.inRange(imagehsv, lower_dusk_sky, upper_dusk_sky)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, 255, 5])
    imagemask_black = cv2.inRange(imagehsv, lower_black, upper_black)

    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([179, 255, 30])
    imagemask_shadow = cv2.inRange(imagehsv, lower_shadow, upper_shadow)
    imagemask_shadow = 1 - imagemask_shadow / 255

    lower_white_cloud = np.array([0, 0, 255])
    upper_white_cloud = np.array([179, 255, 255])
    imagemask_white_cloud = cv2.inRange(
        imagehsv, lower_white_cloud, upper_white_cloud
    )

    lower_dusk_cloud = np.array([0, 0, 179])
    upper_dusk_cloud = np.array([230, 50, 255])
    imagemask_dusk_cloud = cv2.inRange(
        imagehsv, lower_dusk_cloud, upper_dusk_cloud
    )
    depth_sky_removed = (
        depth_map * (1 - imagemask_sky / 255) * (1 - imagemask_dusk_sky / 255)
    )
    depth_all_removed = (
        depth_sky_removed
        * (1 - imagemask_black / 255)
        * (1 - imagemask_white_cloud / 255)
        * (1 - imagemask_dusk_cloud / 255)
    )
    return depth_sky_removed, depth_all_removed, imagemask_shadow


def depth_img_to_pointcld(depth_img, camera_params, extrinsics_mat):
    """convert from depth map to point cloud"""
    f = camera_params[0]
    cx = camera_params[1]
    cy = camera_params[2]

    pointcld = []
    for v in range(0, depth_img.shape[0]):  # x coordinate in image
        for u in range(0, depth_img.shape[1]):  # y coordinate in image
            zw = depth_img[v, u]
            xw = (u - cx) * zw / f
            yw = (v - cy) * zw / f
            point = [xw, yw, zw, 1]
            if not (xw == 0 and yw == 0 and zw == 0):
                point = np.array(point)
                point_w = np.matmul(
                    np.linalg.inv(extrinsics_mat), np.array(point)
                )
                pointcld.append(point_w)
    return pointcld


def pointcld_to_depth_img(pointcld, shape, camera_params, extrinsics_mat):
    """convert from point cloud to depth map"""
    f = camera_params[0]
    cx = camera_params[1]
    cy = camera_params[2]
    intrinsics_mat = np.identity(4)
    intrinsics_mat[0, 2] = cx
    intrinsics_mat[1, 2] = cy
    intrinsics_mat[0, 0] = f
    intrinsics_mat[1, 1] = f
    depth_img = np.zeros(shape)
    for point in pointcld:
        point2d = np.matmul(extrinsics_mat, point)
        depth = point2d[2]
        point2d = 1 / depth * np.matmul(intrinsics_mat, point2d)
        u_ = int(point2d[0])
        v_ = int(point2d[1])
        if 0 <= u_ < shape[1] and 0 <= v_ < shape[0]:
            depth_img[v_, u_] = depth

    return depth_img


def ectract_images_from_db(db_path: str):
    """Add the spatial locations to images in database."""
    db = COLMAPDatabase.connect(db_path)
    db_dict = {}
    for image_id, name, tx, ty in db.execute(
        "SELECT image_id, name, prior_tx, prior_ty FROM images"
    ):
        db_dict[name] = {"id": image_id, "tx": tx, "ty": ty}
    return db_dict


def filter_static(db_path: str):
    """Add the spatial locations to images in database."""
    db = COLMAPDatabase.connect(db_path)
    db_dict = {}
    tx_prev = 0
    ty_prev = 0
    for image_id, name, tx, ty in db.execute(
        "SELECT image_id, name, prior_tx, prior_ty FROM images"
    ):
        if tx_prev == 0 and ty_prev == 0:
            pass
        else:
            dist = gps_to_m(ty_prev, tx_prev, ty, tx)
            if dist < 0.35:
                continue

        db_dict[name] = {"id": image_id, "tx": tx, "ty": ty}
        tx_prev = tx
        ty_prev = ty
    return db_dict


def get_extrinsics_from_images(images, image_id):
    """Get extrinsics from a image dictionary"""
    tvec = images[image_id].tvec
    qvec = images[image_id].qvec
    rotmat = qvec2rotmat(qvec)
    extrinsics_mat = np.identity(4)
    extrinsics_mat[:3, :3] = rotmat
    extrinsics_mat[:3, -1] = tvec
    return extrinsics_mat


def filter_depth(image_name, depth_map, dense_path, args):
    """Filter the depth map using different filters"""
    # ===== Percentile filter =====
    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth_percentile, args.max_depth_percentile]
    )
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    # plot_depth(depth_map, 'original_depth')

    # ===== Range filter =====
    depth_map_range = apply_range_filter(depth_map)
    # plot_depth(depth_map_range, 'range_filtered')

    # ===== HSV color filter =====
    image_path = os.path.join(dense_path, "images")
    raw_image = os.path.join(image_path, image_name)
    (
        depth_map_sky_removed,
        depth_map_all_removed,
        imagemask_shadow,
    ) = apply_hsv_filter(depth_map_range, raw_image)
    # plot_depth(depth_map_sky_removed, 'hsv_filtered')

    return (
        depth_map,
        depth_map_range,
        depth_map_sky_removed,
        depth_map_all_removed,
        imagemask_shadow,
    )


def get_car_mask(seg_mask):
    """Car mask"""
    car_mask = np.ones(seg_mask.shape)
    car_mask[seg_mask == 13] = 0
    car_mask[seg_mask == 14] = 0
    car_mask[seg_mask == 15] = 0
    return car_mask


def get_road_car_mask(seg_mask):
    """Car and road mask"""
    road_car_mask = np.ones(seg_mask.shape)
    road_car_mask[seg_mask == 0] = 0
    road_car_mask[seg_mask == 1] = 0
    road_car_mask[seg_mask == 13] = 0
    road_car_mask[seg_mask == 14] = 0
    road_car_mask[seg_mask == 15] = 0
    return road_car_mask


def get_pedestrain_mask(seg_mask):
    """Pedestrain mask"""
    pedestrain_mask = np.ones(seg_mask.shape)
    pedestrain_mask[seg_mask == 11] = 0
    pedestrain_mask[seg_mask == 12] = 0
    pedestrain_mask[seg_mask == 18] = 0
    return pedestrain_mask


def get_sky_mask(seg_mask):
    """Sky mask"""
    sky_mask = np.ones(seg_mask.shape)
    sky_mask[seg_mask == 10] = 0
    return sky_mask


def parse_args():
    """All Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--postcode", help="Which postcode to run")
    parser.add_argument(
        "--timeofday",
        default="daytime",
        help="Which timeofday to process(daytime, night, dawn_dusk)",
    )
    parser.add_argument(
        "--min_depth_percentile",
        help="minimum visualization depth percentile",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="maximum visualization depth percentile",
        type=float,
        default=95,
    )
    args = parser.parse_args()
    return args


def is_progress_finished(progress_path):
    """Check if postprocessing is finished"""
    with open(progress_path) as fp:
        progress = json.load(fp)
    return progress["Finished?"] or len(progress["Left"]) == 0


def progress_get_next(progress_path):
    """Get the next target to process"""
    with open(progress_path) as fp:
        progress = json.load(fp)
    next_element = progress["Left"].pop(0)
    progress["Doing"].append(next_element)
    with open(progress_path, "w") as fp:
        json.dump(progress, fp, indent=2)
    return next_element


def progress_update(progress_path, element):
    """Update successful postprocessing"""
    with open(progress_path) as fp:
        progress = json.load(fp)
    try:
        progress["Doing"].remove(element)
    except KeyError:
        print(f"This element {element} is not in Doing")
        return
    progress["Done"].append(element)
    if len(progress["Left"]) == 0 and len(progress["Doing"]) == 0:
        progress["Finished?"] = True
    with open(progress_path, "w") as fp:
        json.dump(progress, fp, indent=2)


def failure_update(progress_path, element):
    """Update failed postprocessing"""
    with open(progress_path) as fp:
        progress = json.load(fp)
    try:
        progress["Doing"].remove(element)
    except KeyError:
        print(f"This element {element} is not in Doing")
        return
    progress["Failed"].append(element)
    if len(progress["Left"]) == 0 and len(progress["Doing"]) == 0:
        progress["Finished?"] = True
    with open(progress_path, "w") as fp:
        json.dump(progress, fp, indent=2)


def check_processed_depth(depth_maps_path):
    """Check if depth path is valid"""
    result = True
    if not os.path.exists(depth_maps_path):
        result = False
        print(f"Error: Depth map directory {depth_maps_path} is not valid")
        return False
    depth_maps_bin = glob.glob(os.path.join(depth_maps_path, "*.bin"))
    depth_maps_png = glob.glob(os.path.join(depth_maps_path, "*.png"))
    if len(depth_maps_bin) != len(depth_maps_png) or len(depth_maps_png) == 0:
        print(
            f"Warning: Depth map directory {depth_maps_path}",
            " is missing depth maps",
        )
        result = False
        return result
    return result


def postprocess(dense_path, args):
    """Postprocess depth maps for a path"""
    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError(
            "min_depth_percentile should be less than or equal "
            "to the max_depth_percentile."
        )

    target_path = ("/").join(dense_path.split("/")[:-2])
    db_path = os.path.join(target_path, "database.db")
    images_dict = filter_static(db_path)
    image_name_list = [*images_dict.keys()]

    depth_path = os.path.join(dense_path, "stereo", "depth_maps")
    depth_maps_geometric = glob.glob(depth_path + "/*.geometric.bin")
    depth_map_processed_path = os.path.join(dense_path, "depth_maps_processed")
    # Loop all geometric bin
    for depth_map_ in depth_maps_geometric:
        image_name = ".".join(depth_map_.split("/")[-1].split(".")[:2])
        if image_name not in images_dict:
            continue
        # Read depth and normal maps corresponding to the same image.
        if not os.path.exists(depth_map_):
            print(f"File not found: {depth_map_}")
            continue
        depth_map = read_array(depth_map_)

        (
            depth_map,
            _,
            _,
            depth_map_all_removed,
            shadow_mask,
        ) = filter_depth(image_name, depth_map, dense_path, args)

        # ===== Neighbor depth filter =====
        sparse_path = os.path.join(dense_path, "sparse")
        cameras, images, _ = read_model(sparse_path)
        camera_params = cameras[1].params

        image_index = image_name_list.index(image_name)

        if image_index == len(image_name_list) - 1:
            image_name_neighbor = image_name_list[image_index - 1]
        else:
            image_name_neighbor = image_name_list[image_index + 1]
            if image_name_neighbor[:17] != image_name[:17]:
                image_name_neighbor = image_name_list[image_index - 1]

        image_id = images_dict[image_name]["id"]
        image_id_neighbor = images_dict[image_name_neighbor]["id"]

        if image_id_neighbor not in images:
            image_id_neighbor = image_id
        extrinsics_mat_neighbor = get_extrinsics_from_images(
            images, image_id_neighbor
        )

        depth_map_neighbor_ = os.path.join(
            depth_path, image_name_neighbor + ".geometric.bin"
        )
        if not os.path.exists(depth_map_neighbor_):
            print(f"File not found: {depth_map_neighbor_}")
            continue

        depth_map_neighbor = read_array(depth_map_neighbor_)
        (
            depth_map_neighbor,
            _,
            _,
            depth_map_all_removed_neighbor,
            _,
        ) = filter_depth(
            image_name_neighbor, depth_map_neighbor, dense_path, args
        )
        pointcld = depth_img_to_pointcld(
            depth_map_all_removed_neighbor,
            camera_params,
            extrinsics_mat_neighbor,
        )

        extrinsics_mat = get_extrinsics_from_images(images, image_id)
        shape = depth_map_all_removed.shape
        depth_map_reproject = pointcld_to_depth_img(
            pointcld, shape, camera_params, extrinsics_mat
        )

        diff = abs(depth_map_reproject - depth_map_all_removed)
        diff = (
            diff
            * (diff != depth_map_reproject)
            * (diff != depth_map_all_removed)
        )
        diff_percent = np.divide(
            diff,
            depth_map_all_removed,
            out=np.zeros_like(depth_map_all_removed),
            where=depth_map_all_removed != 0,
        )
        diff_mask = np.ones(diff.shape)
        diff_mask[diff_percent > 0.1] = 0

        # Seg mask
        seg_mask_name = image_name[:-4] + ".png"
        seg_mask_path = os.path.join(dense_path, "seg_mask", seg_mask_name)
        if os.path.exists(seg_mask_path):
            seg_mask = np.array(Image.open(seg_mask_path))
        else:
            seg_mask = np.ones(depth_map)
            seg_mask = int(100) * seg_mask

        car_mask = get_car_mask(seg_mask)
        sky_mask = get_sky_mask(seg_mask)
        pedestrain_mask = get_pedestrain_mask(seg_mask)
        road_car_mask = get_road_car_mask(seg_mask)

        dynamic_mask = 1 - (1 - diff_mask) * (1 - car_mask)
        shadow_raod_car_mask = 1 - (1 - shadow_mask) * (1 - road_car_mask)
        depth_map_processed = (
            depth_map_all_removed
            * dynamic_mask
            * sky_mask
            * pedestrain_mask
            * shadow_raod_car_mask
        )

        depth_map_processed_visual_path = os.path.join(
            dense_path, "depth_maps_visual_processed"
        )
        os.makedirs(depth_map_processed_path, exist_ok=True)
        os.makedirs(depth_map_processed_visual_path, exist_ok=True)

        depth_img = Image.fromarray(depth_map_processed).convert("L")
        depth_img.save(
            os.path.join(depth_map_processed_path, seg_mask_name),
            dtype=np.float32,
        )
        plot_depth(
            depth_map_processed,
            os.path.join(depth_map_processed_visual_path, seg_mask_name),
            seg_mask_name,
            False,
        )
        depth_map_processed_file = os.path.join(
            depth_map_processed_path, image_name + ".geometric.bin"
        )
        write_array(
            depth_map_processed.astype(np.float32), depth_map_processed_file
        )

        depth_map_cleaned_path = os.path.join(dense_path, "depth_maps_cleaned")
        os.makedirs(depth_map_cleaned_path, exist_ok=True)
        depth_map_cleaned_path = os.path.join(
            depth_map_cleaned_path, image_name + ".geometric.bin"
        )
        depth_map_cleaned = apply_clean_filter(depth_map, sky_mask)
        write_array(
            depth_map_cleaned.astype(np.float32), depth_map_cleaned_path
        )
    return depth_map_processed_path


def main():
    """Postprocess depth for bdd100k data"""
    target_path = "/srv/beegfs02/scratch/bdd100k/data/sfm/postcode"
    args = parse_args()

    postcode_path = os.path.join(target_path, args.postcode)
    timeofday_path = os.path.join(postcode_path, args.timeofday)
    postprocess_progress = os.path.join(
        timeofday_path, "postprocess_progress.json"
    )

    while not is_progress_finished(postprocess_progress):
        dense_path = progress_get_next(postprocess_progress)
        depth_map_processed_path = postprocess(dense_path, args)
        if check_processed_depth(depth_map_processed_path):
            progress_update(postprocess_progress, dense_path)
        else:
            failure_update(postprocess_progress, dense_path)


if __name__ == "__main__":
    main()
