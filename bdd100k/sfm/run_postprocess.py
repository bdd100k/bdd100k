import argparse
import glob
import time
import json
import os
import pdb
import struct
from typing import List, Optional, Dict, Tuple
import cv2
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pylab as plt
from PIL import Image
from scipy import misc, ndimage
import pickle
from bdd100k.sfm.colmap.read_write_model import (
    qvec2rotmat,
    read_model,
    rotmat2qvec,
    write_model,
)
from bdd100k.sfm.colmap.read_write_dense import read_array, write_array
from scalabel.label.transforms import rle_to_mask
import open3d as o3d
from sklearn.neighbors import NearestNeighbors, KDTree


def plot_depth(
    depth_map: np.ndarray,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
):
    """Visualize depth map"""
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
        plt.imsave(save_path, depth_map_visual, cmap=cmap)
    plt.close()


def fit_plane(pcd_array: np.ndarray):
    """Use ransac to fit the ground points to a plane"""
    pcd_array = np.delete(pcd_array, 3, 0)
    pcd_array = pcd_array.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    pdb.set_trace()
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=1, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # Uncomment for visualization
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


def depth_to_pcd(
    depth_img: np.ndarray, camera_params, extrinsics_mat: np.ndarray
):
    """Map from depth map to point cloud"""
    f = camera_params[0]
    cx = camera_params[2]
    cy = camera_params[3]
    width = depth_img.shape[1]
    height = depth_img.shape[0]
    xw = np.tile(list(range(width)), (height, 1)) - cx
    yw = np.tile(list(range(height)), (width, 1)).T - cy
    point_x = (xw * depth_img).reshape(width * height) / f
    point_y = (yw * depth_img).reshape(width * height) / f
    point_z = depth_img.reshape(width * height)
    point = np.stack((point_x, point_y, point_z))
    point = point[:, ~np.all(point == 0, axis=0)]
    point = np.vstack((point, np.ones(point.shape[1])))
    pcd_array = np.matmul(np.linalg.inv(extrinsics_mat), point)
    pcd_list = list(pcd_array.T)
    return pcd_list, pcd_array


def pcd_2_depth(pcd, shape, camera_params, extrinsics_mat):
    """Render depth image at a different extrinsics
    Returns:
        depth_img: the rendered depth image
        pcd_indices: corresponding index of point in pcd list
    """
    f = camera_params[0]
    cx = camera_params[2]
    cy = camera_params[3]
    intrinsics_mat = np.identity(4)
    intrinsics_mat[0, 2] = cx
    intrinsics_mat[1, 2] = cy
    intrinsics_mat[0, 0] = f
    intrinsics_mat[1, 1] = f
    depth_img = np.zeros(shape)
    pcd_indices = np.ones(shape) * -1
    for point_index, point in enumerate(pcd):
        # If depth value is already filtered, skip it
        if all(point == [0.0, 0.0, 0.0, 0.0]):
            continue
        z = point[2]
        point2d = np.matmul(extrinsics_mat, point)
        depth = point2d[2]
        point2d = 1 / depth * np.matmul(intrinsics_mat, point2d)
        u_ = round(point2d[0])
        v_ = round(point2d[1])
        if 0 <= u_ < shape[1] and 0 <= v_ < shape[0]:
            # If there is already a closer depth value at pixel, do not update
            if 0 < depth_img[v_, u_] < depth:
                continue
            depth_img[v_, u_] = depth
            pcd_indices[v_, u_] = point_index
    return depth_img, pcd_indices


def images_2_dict(images: List):
    """Generate a dictionary of images for easier post processing"""
    images_list = []
    images_dict = {}
    tvec_list = []
    for key, value in images.items():
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


def get_extrinsics_from_images(images, image_id):
    """Get extrinsics from images"""
    tvec = images[image_id].tvec
    qvec = images[image_id].qvec
    R = qvec2rotmat(qvec)
    extrinsics_mat = np.identity(4)
    extrinsics_mat[:3, :3] = R
    extrinsics_mat[:3, -1] = tvec
    return extrinsics_mat


def apply_range_filter(depth_map: np.array, args) -> np.array:
    """Filter depth based on range."""
    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError(
            "min_depth_percentile should be less than or equal "
            "to the max_depth_percentile."
        )
    depth_map_range = depth_map.copy()
    # Percentile filter
    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth_percentile, args.max_depth_percentile]
    )
    depth_map_range[depth_map < min_depth] = min_depth
    depth_map_range[depth_map > max_depth] = max_depth
    depth_map_range[depth_map < args.min_depth] = 0.0
    depth_map_range[depth_map > args.max_depth] = 0.0
    return depth_map_range


def apply_instance_filter(
    depth_map: np.array, pan_seg_dict: Dict, image_name: str
) -> np.array:
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
    depth_map: np.array, pan_seg_dict: Dict, image_name: str
) -> np.array:
    """Fit a plane to the depth map for ground pixels using ransac"""
    depth_map_ground = depth_map.copy()
    ground_mask = np.zeros(depth_map.shape)
    if len(pan_seg_dict[image_name]["sidewalk"]) != 0:
        ground_mask += pan_seg_dict[image_name]["sidewalk"][0]
    if len(pan_seg_dict[image_name]["ground"]) != 0:
        ground_mask += pan_seg_dict[image_name]["ground"][0]
    if len(pan_seg_dict[image_name]["road"]) != 0:
        ground_mask += pan_seg_dict[image_name]["road"][0]
    depth_ground = depth_map_ground * ground_mask
    return depth_ground


def apply_ground_ransac_filter(
    depth_map: np.array, depth_ground: np.array, camera_params, extrinsics_mat
) -> np.array:
    """Use ransac to fit the ground points to a plane"""
    depth_map_ground_ransac = depth_map.copy()
    pcd_ground, pcd_ground_array = depth_to_pcd(
        depth_ground,
        camera_params,
        extrinsics_mat,
    )
    pcd_ground_filtered = fit_plane(pcd_ground_array)
    depth_ground_filtered, _ = pcd_2_depth(
        pcd_ground_filtered, depth_map.shape, camera_params, extrinsics_mat
    )
    depth_map_ground_ransac = (
        depth_map_ground_ransac - depth_ground + depth_ground_filtered
    )
    return depth_map_ground_ransac


def create_pan_mask_dict(
    pan_json_path: str, shape=(720, 1280)
) -> Dict[str, np.array]:
    """Create a dictionary for panoptic segmentation"""
    if not os.path.exists(pan_json_path):
        return None
    with open(pan_json_path, "rb") as fp:
        fp_content = json.load(fp)
    frames = fp_content["frames"]
    result = dict()
    for frame in frames:
        img_name = frame["name"]
        labels = frame["labels"]
        pan_dict = {
            "person": [],
            "rider": [],
            "bicycle": [],
            "bus": [],
            "car": [],
            "caravan": [],
            "motorcycle": [],
            "trailer": [],
            "train": [],
            "truck": [],
            "dynamic": [],
            "ego vehicle": [],
            "ground": [],
            "static": [],
            "parking": [],
            "rail track": [],
            "road": [],
            "sidewalk": [],
            "bridge": [],
            "building": [],
            "fence": [],
            "garage": [],
            "guard rail": [],
            "tunnel": [],
            "wall": [],
            "banner": [],
            "billboard": [],
            "lane divider": [],
            "parking sign": [],
            "pole": [],
            "polegroup": [],
            "street light": [],
            "traffic cone": [],
            "traffic device": [],
            "traffic light": [],
            "traffic sign": [],
            "traffic sign frame": [],
            "terrain": [],
            "vegetation": [],
            "sky": [],
            "unlabeled": [],
            "total": None,
        }
        sem_id = {
            "person": 31,
            "rider": 32,
            "bicycle": 33,
            "bus": 34,
            "car": 35,
            "caravan": 36,
            "motorcycle": 37,
            "trailer": 38,
            "train": 39,
            "truck": 40,
            "dynamic": 1,
            "ego vehicle": 2,
            "ground": 3,
            "static": 4,
            "parking": 5,
            "rail track": 6,
            "road": 7,
            "sidewalk": 8,
            "bridge": 9,
            "building": 10,
            "fence": 11,
            "garage": 12,
            "guard rail": 13,
            "tunnel": 14,
            "wall": 15,
            "banner": 16,
            "billboard": 17,
            "lane divider": 18,
            "parking sign": 19,
            "pole": 20,
            "polegroup": 21,
            "street light": 22,
            "traffic cone": 23,
            "traffic device": 24,
            "traffic light": 25,
            "traffic sign": 26,
            "traffic sign frame": 27,
            "terrain": 28,
            "vegetation": 29,
            "sky": 30,
            "unlabeled": 0,
        }
        result[img_name] = pan_dict
        pan_seg_total = np.zeros(shape)
        for label in labels:
            cur_label_mask = rle_to_mask(label["rle"])
            result[img_name][label["category"]].append(cur_label_mask)
            pan_seg_total += cur_label_mask * sem_id[label["category"]]
        result[img_name]["total"] = pan_seg_total
    return result


def parse_args():
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
        "--input-type",
        type=str,
        default="overlaps",
        help="Enter overlaps or singles. One job process either one of two",
    )
    parser.add_argument(
        "--target-path",
        "-t",
        type=str,
        default="/srv/beegfs02/scratch/bdd100k/data/sfm/postcode",
        help="Local path to save all postcodes and the results.",
    )
    parser.add_argument(
        "--dense-path",
        type=str,
        default="",
        help="Dense path to save all postcodes and the results.",
    )
    args = parser.parse_args()
    return args


def prepare_bts_training_data(
    gt: np.array, rgb: np.array, scale: float
) -> Tuple[np.array, np.array]:
    """Normalize the focal length and Crop to the same size as KITTI"""
    # Normalize the focal length based on scale
    height = gt.shape[0]
    width = gt.shape[1]
    gt_normalized = cv2.resize(gt, (0, 0), fx=scale, fy=scale)
    rgb_normalized = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
    # pdb.set_trace()
    new_width, new_height = gt_normalized.shape[1], gt_normalized.shape[0]
    height_margin = int((height - new_height) / 2)
    width_margin = int((width - new_width) / 2)
    gt_new = np.zeros(gt.shape)
    gt_new[
        height_margin : height_margin + new_height,
        width_margin : width_margin + new_width,
    ] = gt_normalized
    rgb_new = np.zeros(rgb.shape)
    rgb_new[
        height_margin : height_margin + new_height,
        width_margin : width_margin + new_width,
    ] = rgb_normalized
    # Crop to the size of KITTI dataset
    # The kept image is closer to the top
    top_margin = int((height - 352) / 4)
    left_margin = int((width - 1216) / 2)
    gt_new = gt_new[
        top_margin : top_margin + 352, left_margin : left_margin + 1216
    ]
    rgb_new = rgb_new[
        top_margin : top_margin + 352, left_margin : left_margin + 1216
    ]
    return gt_new, rgb_new


def postprocess(dense_path, target_path, args):
    """Filter the depth maps through several filters"""
    images_path = os.path.join(target_path, "images")
    sparse_path = os.path.join(dense_path, "sparse")
    cameras, images, points3D = read_model(sparse_path)
    camera_params = cameras[1].params
    images_dict = images_2_dict(images)
    depth_path = os.path.join(dense_path, "stereo", "depth_maps")
    depth_maps_processed_path = os.path.join(
        dense_path, "depth_maps_processed"
    )
    depth_map_processed_visual_path = os.path.join(
        dense_path, "depth_maps_visual_processed"
    )
    bts_train_rgb_path = os.path.join(dense_path, "bts_train", "rgb")
    bts_train_gt_path = os.path.join(dense_path, "bts_train", "gt")

    os.makedirs(depth_maps_processed_path, exist_ok=True)
    os.makedirs(depth_map_processed_visual_path, exist_ok=True)
    os.makedirs(bts_train_rgb_path, exist_ok=True)
    os.makedirs(bts_train_gt_path, exist_ok=True)
    depth_maps_geometric = glob.glob(depth_path + "/*.geometric.bin")

    pan_seg_path = os.path.join(target_path, "pan_mask", "pan_seg.json")
    pan_seg_dict = create_pan_mask_dict(pan_seg_path)

    # Loop all geometric bin
    for depth_map_ in depth_maps_geometric:
        image_name = ".".join(depth_map_.split("/")[-1].split(".")[:2])
        image_name_png = image_name[:-4] + ".png"
        if not image_name in images_dict:
            print("File not found: {} in database".format(depth_map_))
            continue
        if not os.path.exists(depth_map_):
            print("File not found: {}".format(depth_map_))
            continue
        image_id = images_dict[image_name]["id"]
        extrinsics_mat = get_extrinsics_from_images(images, image_id)

        # Read depth and normal maps corresponding to the same image.
        depth_map = read_array(depth_map_)

        # Apply range filter
        depth_map_range = apply_range_filter(depth_map, args)

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

        # Save depth map
        # Save depth to the second decimal in unit16
        depth_map_uint16 = (depth_map_processed * 100).astype(np.uint16)
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

        # Prepare and save data for BTS depth training
        # Mainly focal length normalization
        hi = depth_map_processed.shape[0]
        wi = depth_map_processed.shape[1]
        depth_density = len(depth_map_processed[depth_map_processed != 0]) / (
            hi * wi
        )
        # Only consider depth maps with depth density > 30% for training
        if depth_density > 0.1:
            rgb_path = os.path.join(images_path, image_name)
            rgb = cv2.imread(rgb_path)
            gt = depth_map_uint16
            FOCAL_KITTI = 715.0873
            focal_scale = FOCAL_KITTI / float(camera_params[0])
            gt_new, rgb_new = prepare_bts_training_data(gt, rgb, focal_scale)
            cv2.imwrite(
                os.path.join(bts_train_gt_path, image_name_png),
                gt_new.astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join(bts_train_rgb_path, image_name_png), rgb_new
            )
    return depth_maps_processed_path


def main():
    args = parse_args()
    target_path = args.target_path
    dense_path = args.dense_path
    depth_maps_processed_path = postprocess(dense_path, target_path, args)


if __name__ == "__main__":
    main()
