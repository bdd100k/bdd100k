"""Scripts for postprocessing COLMAP depth data."""
import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scalabel.common.typing import NDArrayF64, NDArrayU8

try:
    import cv2
    import open3d as o3d
    from sklearn.neighbors import KDTree
except ImportError:
    pass

from bdd100k.sfm.colmap.read_write_dense import read_array
from bdd100k.sfm.colmap.read_write_model import (
    Camera,
    Image,
    qvec2rotmat,
    read_model,
)

from .utils import create_pan_mask_dict, depth_to_pcd, pcd_to_depth, plot_depth

# Camera Intrinsics of KITTI-360
FOCAL_KITTI = 715.0873
H_KITTI = 352
W_KITTI = 1216


"""Arguments."""
parser = argparse.ArgumentParser(description="Postprocess depth map")
parser.add_argument(
    "--dense_path",
    type=str,
    default="",
    help="Dense path to save all postcodes and the results.",
)
parser.add_argument(
    "--target_path",
    "-t",
    type=str,
    default="postcode",
    help="Local path to save all postcodes and the results.",
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
parser.add_argument(
    "--min_depth",
    help="minimum visualization depth in meter",
    type=float,
    default=3,
)
parser.add_argument(
    "--max_depth",
    help="maximum visualization depth in meter",
    type=float,
    default=80,
)
parser.add_argument(
    "--crop_bts",
    action="store_true",
    help="Crop images for bts training.",
)
ARGUMENTS = parser.parse_args()


def fit_plane(pcd_array: NDArrayF64) -> List[NDArrayF64]:
    """Use ransac to fit the points of floor/ground to a plane.

    This is used to avoid noise in depth map for the ground.
    """
    pcd_array = np.delete(pcd_array, 3, 0)
    pcd_array = pcd_array.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    if len(pcd_array) < 10:
        # If there are less than 10 pixels of ground
        # remove all of them since ransac cannot work well
        return []
    _, inliers = pcd.segment_plane(
        distance_threshold=1, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    # Uncomment for pcd visualization
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                             zoom=0.8,
    #                             front=[-0.4999, -0.1659, -0.8499],
    #                             lookat=[2.1813, 2.0619, 2.0999],
    #                             up=[0.1204, -0.9852, 0.1215])
    pcd_ground_inlier = []
    for index in inliers:
        point = np.ones(4)
        point[:3] = pcd_array[index]
        pcd_ground_inlier.append(point)
    return pcd_ground_inlier


def images_to_dict(images: Dict[int, Image]) -> Dict[str, Dict[str, int]]:
    """Generate a dictionary of images for easier post processing.

    output:
    images_dict: key is the image name, value is a dict containing id, tvec,
    neighbors, depth_map and depth_map_nbr.
    """
    images_list = []
    images_dict = {}
    tvec_list = []
    for _, value in images.items():
        images_list.append(value)
        tvec_list.append(value.tvec)
        images_dict[value.name] = {
            "id": value.id,
            "tvec": value.tvec,
            "neighbors": [],
            "depth_map": None,
            "depth_map_nbr": None,
        }
    tvec_array = np.array(tvec_list)
    kdt = KDTree(tvec_array, leaf_size=30, metric="euclidean")
    dists, indexes = kdt.query(tvec_array, k=6, return_distance=True)
    for i, image in enumerate(images_list):
        for j in range(1, 6):
            images_dict[image.name]["neighbors"].append(
                (images_list[indexes[i][j]].name, dists[i][j])
            )
    return images_dict


def get_extrinsics_from_images(
    images: Dict[int, Image], image_id: int
) -> NDArrayF64:
    """Get extrinsics from images."""
    tvec = images[image_id].tvec
    qvec = images[image_id].qvec
    rot_mat = qvec2rotmat(qvec)
    extrinsics_mat = np.identity(4)
    extrinsics_mat[:3, :3] = rot_mat
    extrinsics_mat[:3, -1] = tvec
    return extrinsics_mat


def apply_range_filter(depth_map: NDArrayF64) -> NDArrayF64:
    """Filter depth based on range."""
    if ARGUMENTS.min_depth_percentile > ARGUMENTS.max_depth_percentile:
        raise ValueError(
            "min_depth_percentile should be less than or equal "
            "to the max_depth_percentile."
        )
    depth_map_range = depth_map.copy()
    # Percentile filter
    min_depth, max_depth = np.percentile(
        depth_map,
        [ARGUMENTS.min_depth_percentile, ARGUMENTS.max_depth_percentile],
    )
    depth_map_range[depth_map < min_depth] = min_depth
    depth_map_range[depth_map > max_depth] = max_depth
    depth_map_range[depth_map < ARGUMENTS.min_depth] = 0.0
    depth_map_range[depth_map > ARGUMENTS.max_depth] = 0.0
    return depth_map_range


def apply_instance_filter(
    depth_map: NDArrayF64,
    pan_seg_dict: Optional[Dict[str, Dict[str, List[NDArrayU8]]]],
    image_name: str,
) -> NDArrayF64:
    """Filter depth based on instance segmentation."""
    depth_map_instance = depth_map.copy()
    instances_mask = (
        pan_seg_dict[image_name]["car"]
        + pan_seg_dict[image_name]["bus"]
        + pan_seg_dict[image_name]["truck"]
        + pan_seg_dict[image_name]["caravan"]
    )
    if len(instances_mask) != 0:
        for instance_mask in instances_mask:
            instance_depth_img = instance_mask * depth_map_instance
            instance_depth = instance_depth_img[instance_depth_img != 0]
            if len(instance_depth) == 0:
                continue

            median = np.median(instance_depth)
            std = np.std(instance_depth)
            max_thres = median + 2 * std
            min_thres = median - 2 * std
            # if the standard deviation is greater than 4 meter
            # remove the whole instance
            depth_map_instance = depth_map_instance * (1 - instance_mask)
            if std < 4:
                depth_map_instance += (
                    instance_depth_img
                    * (instance_depth_img > min_thres)
                    * (instance_depth_img < max_thres)
                )
    for stuff_class in ["unlabeled", "ego vehicle", "sky", "dynamic"]:
        if len(pan_seg_dict[image_name][stuff_class]) > 0:
            depth_map_instance *= 1 - pan_seg_dict[image_name][stuff_class][0]

    for thing_class in ["person", "rider", "bicycle", "motorcycle"]:
        if len(pan_seg_dict[image_name][thing_class]) > 0:
            for ins in pan_seg_dict[image_name][thing_class]:
                depth_map_instance *= 1 - ins
    return depth_map_instance


def get_ground_depth(
    depth_map: NDArrayF64,
    pan_seg_dict: Optional[Dict[str, Dict[str, List[NDArrayU8]]]],
    image_name: str,
) -> NDArrayF64:
    """Fit a plane to the depth map for ground pixels using ransac."""
    depth_map_ground = depth_map.copy()
    ground_mask = np.zeros(depth_map.shape)
    if len(pan_seg_dict[image_name]["sidewalk"]) != 0:
        ground_mask += pan_seg_dict[image_name]["sidewalk"][0]
    if len(pan_seg_dict[image_name]["ground"]) != 0:
        ground_mask += pan_seg_dict[image_name]["ground"][0]
    if len(pan_seg_dict[image_name]["road"]) != 0:
        ground_mask += pan_seg_dict[image_name]["road"][0]
    depth_ground: NDArrayF64 = depth_map_ground * ground_mask
    return depth_ground


def apply_ground_ransac_filter(
    depth_map: NDArrayF64,
    depth_ground: NDArrayF64,
    camera_params: Camera,
    extrinsics_mat: NDArrayF64,
) -> NDArrayF64:
    """Use ransac to fit the ground points to a plane."""
    depth_map_ground_ransac = depth_map.copy()
    pcd_ground_array = depth_to_pcd(
        depth_ground,
        camera_params,
        extrinsics_mat,
    )
    pcd_ground_filtered = fit_plane(pcd_ground_array)
    depth_ground_filtered, _ = pcd_to_depth(
        pcd_ground_filtered, depth_map.shape, camera_params, extrinsics_mat
    )
    depth_map_ground_ransac = (
        depth_map_ground_ransac - depth_ground + depth_ground_filtered
    )
    return depth_map_ground_ransac


def prepare_bts_training_data(
    g_t: NDArrayF64, rgb: NDArrayU8, focal: float
) -> Tuple[NDArrayF64, NDArrayF64]:
    """Prepare depth dataset to the same format as KITTI for BTS training.

    KITTI intrinsics: H: 352, W: 1216, f: 715.0873

    Method fn (focal length normalization):
        Normalize the focal length and Crop to the same size as KITTI
    """
    height = g_t.shape[0]
    width = g_t.shape[1]
    scale = FOCAL_KITTI / focal
    top_margin = int((height - H_KITTI) / 2)
    left_margin = int((width - W_KITTI) / 2)

    # Normalize the focal length based on scale
    gt_normalized = cv2.resize(
        g_t, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    rgb_normalized = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
    new_width, new_height = gt_normalized.shape[1], gt_normalized.shape[0]
    height_margin = int((height - new_height) / 2)
    width_margin = int((width - new_width) / 2)
    gt_fn = np.zeros(g_t.shape)
    gt_fn[
        height_margin : height_margin + new_height,
        width_margin : width_margin + new_width,
    ] = gt_normalized
    rgb_fn = np.zeros(rgb.shape)
    rgb_fn[
        height_margin : height_margin + new_height,
        width_margin : width_margin + new_width,
    ] = rgb_normalized
    gt_fn = gt_fn[
        top_margin : top_margin + H_KITTI, left_margin : left_margin + W_KITTI
    ]
    rgb_fn = rgb_fn[
        top_margin : top_margin + H_KITTI, left_margin : left_margin + W_KITTI
    ]
    return gt_fn, rgb_fn


def postprocess(dense_path: str, target_path: str) -> str:
    """Filter the depth maps through several filters."""
    images_path = os.path.join(target_path, "images")
    sparse_path = os.path.join(dense_path, "sparse")
    cameras, images, _ = read_model(sparse_path)
    camera_params = cameras[1].params
    images_dict = images_to_dict(images)
    depth_path = os.path.join(dense_path, "stereo", "depth_maps")
    depth_maps_processed_path = os.path.join(
        dense_path, "depth_maps_processed"
    )
    depth_map_processed_visual_path = os.path.join(
        dense_path, "depth_maps_visual_processed"
    )

    os.makedirs(depth_maps_processed_path, exist_ok=True)
    os.makedirs(depth_map_processed_visual_path, exist_ok=True)
    depth_maps_geometric = glob.glob(depth_path + "/*.geometric.bin")

    pan_seg_path = os.path.join(target_path, "pan_mask", "pan_seg.json")
    pan_seg_dict = create_pan_mask_dict(pan_seg_path)

    # Loop all geometric bin
    for depth_map_ in depth_maps_geometric:
        image_name = ".".join(depth_map_.split("/")[-1].split(".")[:2])
        image_name_png = image_name[:-4] + ".png"
        if image_name not in images_dict:
            print(f"File not found: {depth_map_} in database")
            continue
        if not os.path.exists(depth_map_):
            print(f"File not found: {depth_map_}")
            continue
        image_id = images_dict[image_name]["id"]
        extrinsics_mat = get_extrinsics_from_images(images, image_id)

        # Read depth and normal maps corresponding to the same image.
        depth_map = read_array(depth_map_)

        # Apply range filter
        depth_map_range = apply_range_filter(depth_map)

        # Apply instance segmentation mask filter
        depth_map_instance = apply_instance_filter(
            depth_map_range, pan_seg_dict, image_name
        )

        # Apply plane fitting filter
        depth_ground = get_ground_depth(
            depth_map_instance, pan_seg_dict, image_name
        )
        depth_map_ground_ransac = apply_ground_ransac_filter(
            depth_map_instance, depth_ground, camera_params, extrinsics_mat
        )

        depth_map_processed = depth_map_ground_ransac

        # Prepare and save data for BTS depth training
        # Mainly focal length normalization
        hi = depth_map_processed.shape[0]
        wi = depth_map_processed.shape[1]
        depth_density = len(depth_map_processed[depth_map_processed != 0]) / (
            hi * wi
        )

        if depth_density > 0.1:
            # Save depth map
            # Save depth to the second decimal in unit16
            depth_map_processed_save: NDArrayF64 = depth_map_processed * 256
            depth_map_uint16 = depth_map_processed_save.astype(np.uint16)
            cv2.imwrite(
                os.path.join(depth_maps_processed_path, image_name_png),
                depth_map_uint16,
            )
            plot_depth(
                depth_map_processed,
                os.path.join(depth_map_processed_visual_path, image_name_png),
                image_name_png,
                False,
            )

        # Only consider depth maps with depth density > 10% for training
        if ARGUMENTS.crop_bts:
            bts_train_rgb_fn_path = os.path.join(
                dense_path, "bts_train_fn", "rgb"
            )
            bts_train_gt_fn_path = os.path.join(
                dense_path, "bts_train_fn", "gt"
            )
            os.makedirs(bts_train_rgb_fn_path, exist_ok=True)
            os.makedirs(bts_train_gt_fn_path, exist_ok=True)
            if depth_density > 0.1:
                rgb_path = os.path.join(images_path, image_name)
                rgb = cv2.imread(rgb_path)
                gt = depth_map_uint16
                gt_fn, rgb_fn = prepare_bts_training_data(
                    gt, rgb, float(camera_params[0])
                )
                cv2.imwrite(
                    os.path.join(bts_train_gt_fn_path, image_name_png),
                    gt_fn.astype(np.uint16),
                )
                cv2.imwrite(
                    os.path.join(bts_train_rgb_fn_path, image_name_png), rgb_fn
                )
    return depth_maps_processed_path


def main() -> None:
    """Post process depth maps."""
    target_path = ARGUMENTS.target_path
    dense_path = ARGUMENTS.dense_path
    postprocess(dense_path, target_path)


if __name__ == "__main__":
    main()
