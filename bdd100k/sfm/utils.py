"""SfM / GPS info utilities."""
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
    from geopy.location import Location
except ImportError:
    pass
import matplotlib.pyplot as plt
from matplotlib import cm
from scalabel.common.typing import (
    DictStrAny,
    NDArrayF64,
    NDArrayI32,
    NDArrayU8,
)
from scalabel.label.transforms import rle_to_mask
from scalabel.label.typing import Extrinsics, Frame, Intrinsics

from bdd100k.sfm.colmap.read_write_dense import read_array  # type: ignore


def cam_spec_prior(
    intrinsics_path: Optional[str] = "",
) -> Optional[Intrinsics]:
    """Generate intrinsics from iPhone 5 cam spec prior."""
    if intrinsics_path == "":
        # Empty string means no intrinsics are given
        intrinsics = None
    elif intrinsics_path == "bdd100k":
        # For bdd100 sequences
        # 4.1 (F in mm) / 0.0014 (pixel size in mm) = 1020 focal length in
        # pixels
        # resolution always (720, 1280) --> principal point at (360, 640)
        # These are intrinsics estimated for bdd100k
        intrinsics = Intrinsics(focal=(1020.0, 1020.0), center=(360.0, 640.0))
    else:
        if intrinsics_path:
            with open(intrinsics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            intrinsics = Intrinsics(
                focal=(data["fx"], data["fy"]), center=(data["cy"], data["cx"])
            )
        else:
            intrinsics_path = None
    return intrinsics


def latlon_from_data(data: List[DictStrAny]) -> NDArrayF64:
    """Return an Nx2 array of latitude / longitude from given list of data."""
    return np.array([[d["latitude"], d["longitude"]] for d in data])


def get_location_from_data(data: DictStrAny) -> Location:
    """Get location from location data."""
    geolocator = Nominatim(user_agent="application")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    location = reverse(
        (data["latitude"], data["longitude"]), language="en", exactly_one=True
    )
    address = location.raw["address"]
    return address


def latlon_to_cartesian(lat: float, lon: float) -> Tuple[float, float, float]:
    """Generate 3D dim cartesian coordinates from longitude / latitude."""
    r = 6371008.7714  # https://en.wikipedia.org/wiki/Earth_radius
    theta = math.pi / 2 - math.radians(lat)
    phi = math.radians(lon)
    x = r * math.sin(theta) * math.cos(phi)  # bronstein (3.381a)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def load_pose_data(info_path: str) -> Optional[List[DictStrAny]]:
    """Load pose data for given sequences from the sequence info file."""
    if os.stat(info_path).st_size:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)["locations"]
        return data  # type: ignore
    return None


def frames_from_images(
    image_path: str, seq_name: Optional[str] = None
) -> List[Frame]:
    """Construct a list of Frame objects from an input image path."""
    if not seq_name:
        images = sorted(os.listdir(image_path))
        frames = [
            Frame(name=im, videoName=seq_name, frameIndex=i)
            for i, im in enumerate(images)
        ]
    else:
        images = glob.glob(os.path.join(image_path, f"{seq_name}*"))
        images = sorted([os.path.split(image)[1] for image in images])
        frames = [
            Frame(name=im, videoName=seq_name, frameIndex=i)
            for i, im in enumerate(images)
        ]
    return frames


def get_poses_from_data(list_data: List[DictStrAny]) -> NDArrayF64:
    """Generate 6 DoF poses from GPS location data."""
    # convention: right handed coord system, x forward, y right, z up
    x0, y0, _ = latlon_to_cartesian(
        list_data[0]["latitude"], list_data[0]["longitude"]
    )
    locations = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    for data in list_data[1:]:
        x, y, _ = latlon_to_cartesian(data["latitude"], data["longitude"])
        yaw = np.deg2rad(data["course"] - list_data[0]["course"])
        pitch, roll = 0.0, 0.0
        locations.append([x - x0, y - y0, 0.0, pitch, roll, yaw])
    return np.array(locations)


def get_gps_from_data(list_data: List[DictStrAny]) -> NDArrayF64:
    """Generate GPS location data."""
    locations = []
    for data in list_data:
        locations.append(
            [data["latitude"], data["longitude"], 0.0, 0.0, 0.0, 0.0]
        )
    return np.array(locations)


def interpolate_trajectory(
    gps_prior: List[DictStrAny],
    frames: List[Frame],
    intrinsics_path: Optional[str] = "",
) -> None:
    """Interpolate GPS based pose priors to per frame poses."""
    num_frames_per_pose = len(frames) / len(gps_prior)
    gps_poses = get_poses_from_data(gps_prior)

    for i, f in enumerate(frames):
        f.intrinsics = cam_spec_prior(intrinsics_path)

        pose_index = int(i / num_frames_per_pose)
        weight_hi = i / num_frames_per_pose - i // num_frames_per_pose

        traj_lo = np.array(gps_poses[pose_index])
        if len(gps_poses) - 1 > pose_index:
            traj_hi = np.array(gps_poses[pose_index + 1])
            traj_cur: NDArrayF64 = (
                weight_hi * traj_hi + (1 - weight_hi) * traj_lo
            )
        else:
            traj_hi = traj_lo
            traj_lo = np.array(gps_poses[pose_index - 1])
            velo: NDArrayF64 = traj_hi - traj_lo
            traj_cur = traj_hi + velo * weight_hi

        f.extrinsics = Extrinsics(
            location=tuple(traj_cur[:3]), rotation=tuple(traj_cur[3:])
        )


def interpolate_gps(
    gps_prior: List[DictStrAny],
    frames: List[Frame],
    intrinsics_path: Optional[str] = "",
) -> Tuple[List[Frame], List[Frame]]:
    """Interpolate GPS priors to per frame poses."""
    num_frames_per_pose = len(frames) / len(gps_prior)
    gps_poses = get_gps_from_data(gps_prior)
    frames_filtered = []
    frames_skipped = []
    traj_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i, f in enumerate(frames):
        f.intrinsics = cam_spec_prior(intrinsics_path)

        pose_index = int(i / num_frames_per_pose)
        weight_hi = i / num_frames_per_pose - i // num_frames_per_pose

        traj_lo = np.array(gps_poses[pose_index])
        if len(gps_poses) - 1 > pose_index:
            traj_hi: NDArrayF64 = np.array(gps_poses[pose_index + 1])
            traj_cur = weight_hi * traj_hi + (1 - weight_hi) * traj_lo
        else:
            traj_hi = traj_lo
            traj_lo = np.array(gps_poses[pose_index - 1])
            velo: NDArrayF64 = traj_hi - traj_lo
            traj_cur = traj_hi + velo * weight_hi

        f.extrinsics = Extrinsics(
            location=tuple(traj_cur[:3]), rotation=tuple(traj_cur[3:])
        )
        dist_moved = gps_to_m(
            traj_prev[0],
            traj_prev[1],
            traj_cur[0],
            traj_cur[1],
        )
        traj_prev = traj_cur
        if i > 0 and dist_moved < 0.1:
            frames_skipped.append(f)
            continue
        frames_filtered.append(f)
    return frames_filtered, frames_skipped


def get_pose_priors(info_path: str, image_path: str) -> Optional[List[Frame]]:
    """Generate Scalabel frames with pose priors from gps / image paths."""
    pose_data = load_pose_data(info_path)
    if pose_data is not None:
        frames = frames_from_images(image_path)
        interpolate_trajectory(pose_data, frames)
        return frames
    return None


def get_gps_priors(
    info_path_list: List[str],
    image_path: str,
    intrinsics_path: Optional[str] = "",
) -> List[Frame]:
    """Generate Scalabel frames with gps priors from gps / image paths."""
    frames: List[Frame] = []
    frames_to_move: List[Frame] = []
    # if there is already skiped images, we reput them into image folder
    # and redo the whole precess
    dir_name = os.path.dirname(image_path)
    skipped_image_path = os.path.join(dir_name, "images_skipped")
    if os.path.exists(skipped_image_path):
        skipped_image_path_all = os.path.join(skipped_image_path, "*")
        os.system(f"mv {skipped_image_path_all} {image_path}")
        os.system(f"rm -r {skipped_image_path}")

    for info_path in info_path_list:
        pose_data = load_pose_data(info_path)
        seq_name = os.path.splitext(os.path.basename(info_path))[0]
        if pose_data is not None:
            cur_frames = frames_from_images(image_path, seq_name)
            interpolated_frames, skipped_frames = interpolate_gps(
                pose_data, cur_frames, intrinsics_path
            )
            frames += interpolated_frames
            frames_to_move += skipped_frames
    remove_skipped_frames(image_path, frames_to_move)
    return frames


def remove_skipped_frames(
    image_path: str, skipped_frames: List[Frame]
) -> None:
    """Move skipped frames from image folder to a new folder."""
    dir_name = os.path.dirname(image_path)
    skipped_image_path = os.path.join(dir_name, "images_skipped")
    if os.path.exists(skipped_image_path):
        return
    os.system(f"mkdir {skipped_image_path}")
    for frame in skipped_frames:
        cur_image = os.path.join(image_path, frame.name)
        try:
            os.system(f"mv {cur_image} {skipped_image_path}")
        except FileNotFoundError:
            print(f"{frame.name} is not in {image_path}")


def gps_to_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Estimate metric distance between 2 gps coords."""
    radius = 6378.137  # Radius of earth in KM
    dlat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dlon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(
        lat1 * np.pi / 180
    ) * np.cos(lat2 * np.pi / 180) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = float(radius * c * 1000.0)
    return d


def create_pan_mask_dict(
    pan_json_path: str, shape: Tuple[int, int] = (720, 1280)
) -> Optional[Dict[str, Dict[str, List[NDArrayU8]]]]:
    """Create a dictionary for panoptic segmentation."""
    if not os.path.exists(pan_json_path):
        return None
    with open(pan_json_path, "rb") as fp:
        fp_content = json.load(fp)
    frames = fp_content["frames"]
    result = {}
    for frame in frames:
        img_name = frame["name"]
        labels = frame["labels"]
        pan_dict: Dict[str, List[NDArrayU8]] = {
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
            "total": [],
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
        result[img_name]["total"] = [pan_seg_total]  # type: ignore
    return result


def depth_to_pcd(
    depth_img: NDArrayF64,
    camera_params: NDArrayF64,
    extrinsics_mat: NDArrayF64,
) -> NDArrayF64:
    """Map from 2D depth map to 3D point cloud."""
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
    pcd_array = np.array(np.matmul(np.linalg.inv(extrinsics_mat), point))
    return pcd_array


def pcd_to_depth(
    pcd: List[NDArrayF64],
    shape: Tuple[int, int],
    camera_params: NDArrayF64,
    extrinsics_mat: NDArrayF64,
) -> Tuple[NDArrayF64, NDArrayI32]:
    """Map from 3D point cloud to 2D depth map at a different extrinsics.

    outputs:
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
    pcd_indices: NDArrayI32 = np.ones(shape) * -1  # type: ignore
    for point_index, point in enumerate(pcd):
        # If depth value is already filtered, skip it
        if all(point == [0.0, 0.0, 0.0, 0.0]):
            continue
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


def plot_depth(
    depth_map: NDArrayF64,
    save_path: str = "",
    title: str = "",
    visualize: bool = True,
    vmin: float = 0,
    vmax: float = 80,
) -> None:
    """Visualize depth map."""
    mask = np.logical_or((depth_map == vmin), (depth_map > vmax))
    depth_map_visual = np.ma.masked_where(mask, depth_map)
    cmap = cm.viridis
    cmap.set_bad(color="gray")
    plt.figure(figsize=(30, 20))
    plt.imshow(depth_map_visual, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)

    if visualize:
        plt.show()
    else:
        plt.imsave(save_path, depth_map_visual, cmap=cmap)
    plt.close()


def get_fusion_mask_pan(
    depth_map: NDArrayF64, pan_seg_dict: Optional[Dict[str, List[NDArrayU8]]]
) -> NDArrayU8:
    """Update the image masks for stereo fusion."""
    mask_fusion = np.ones(depth_map.shape, dtype=np.int8)
    # Percentile filter
    min_depth, max_depth = np.percentile(depth_map, [5, 95])
    mask_fusion[depth_map < min_depth] = 0
    mask_fusion[depth_map > max_depth] = 0
    # Only keep of points within 3 - 450 meter
    mask_fusion[depth_map < 3] = 0
    mask_fusion[depth_map > 450] = 0

    pan_mask = np.zeros(depth_map.shape)
    if pan_seg_dict is not None:
        if len(pan_seg_dict["ego vehicle"]) != 0:
            pan_mask += pan_seg_dict["ego vehicle"][0]
        if len(pan_seg_dict["sky"]) != 0:
            pan_mask += pan_seg_dict["sky"][0]
        transient_instances = (
            pan_seg_dict["car"]
            + pan_seg_dict["bus"]
            + pan_seg_dict["truck"]
            + pan_seg_dict["person"]
            + pan_seg_dict["rider"]
            + pan_seg_dict["bicycle"]
        )
        if len(transient_instances) != 0:
            for instance_mask in transient_instances:
                pan_mask += instance_mask
        pan_mask = 1 - pan_mask
        mask_fusion = mask_fusion * pan_mask  # type: ignore
    return mask_fusion  # type: ignore


def create_fusion_masks_pan(dense_path: str, pan_mask_path: str) -> str:
    """Create image masks only for stereo fusion."""
    print(f"Creating fusion mask for: {dense_path}")
    depth_path = os.path.join(dense_path, "stereo", "depth_maps")
    depth_maps_geometric = glob.glob(depth_path + "/*.geometric.bin")
    fusion_mask_path = os.path.join(dense_path, "fusion_mask")
    pan_mask_path = os.path.join(pan_mask_path, "pan_seg.json")
    pan_mask_dict = create_pan_mask_dict(pan_mask_path)
    os.makedirs(fusion_mask_path, exist_ok=True)
    for depth_map_path in depth_maps_geometric:
        image_name = ".".join(depth_map_path.split("/")[-1].split(".")[:2])

        if not os.path.exists(depth_map_path):
            print(f"File not found: {depth_map_path}")
            continue
        depth_map = read_array(depth_map_path)
        # Create fusion mask
        pan_mask = None
        if pan_mask_dict and image_name in pan_mask_dict:
            pan_mask = pan_mask_dict[image_name]
        mask_fusion = get_fusion_mask_pan(depth_map, pan_mask)
        mask_fusion_save_path = f"{fusion_mask_path}/{image_name}.png"
        cv2.imwrite(mask_fusion.astype(np.uint8), mask_fusion_save_path)
    return fusion_mask_path
