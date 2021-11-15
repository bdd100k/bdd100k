"""SfM / GPS info utilities."""
import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location
from scalabel.common.typing import DictStrAny, NDArrayF64
from scalabel.label.typing import Extrinsics, Frame, Intrinsics


def cam_spec_prior() -> Intrinsics:
    """Generate intrinsics from iPhone 5 cam spec prior."""
    # 4.1 (F in mm) / 0.0014 (pixel size in mm) = 2928.57 focal length in
    # pixels
    # resolution always (720, 1280) --> principal point at (360, 640)
    intrinsics = Intrinsics(focal=(2928.57, 2928.57), center=(360.0, 640.0))
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


def frames_from_images(image_path: str) -> List[Frame]:
    """Construct a list of Frame objects from an input image path."""
    images = sorted(os.listdir(image_path))
    seq_name = os.path.basename(image_path)
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


def interpolate_trajectory(
    gps_prior: List[DictStrAny], frames: List[Frame]
) -> None:
    """Interpolate GPS based pose priors to per frame poses."""
    num_frames_per_pose = len(frames) / len(gps_prior)
    gps_poses = get_poses_from_data(gps_prior)

    for i, f in enumerate(frames):
        f.intrinsics = cam_spec_prior()

        pose_index = int(i / num_frames_per_pose)
        weight_hi = i / num_frames_per_pose - i // num_frames_per_pose

        traj_lo = np.array(gps_poses[pose_index])
        if len(gps_poses) - 1 > pose_index:
            traj_hi: NDArrayF64 = np.array(gps_poses[pose_index + 1])
            traj_cur = weight_hi * traj_hi + (1 - weight_hi) * traj_lo  # type: ignore
        else:
            traj_hi = traj_lo
            traj_lo = np.array(gps_poses[pose_index - 1])
            velo: NDArrayF64 = traj_hi - traj_lo
            traj_cur = traj_hi + velo * weight_hi

        f.extrinsics = Extrinsics(
            location=tuple(traj_cur[:3]), rotation=tuple(traj_cur[3:])
        )


def get_pose_priors(info_path: str, image_path: str) -> Optional[List[Frame]]:
    """Generate Scalabel frames with pose priors from gps / image paths."""
    pose_data = load_pose_data(info_path)
    if pose_data is not None:
        frames = frames_from_images(image_path)
        interpolate_trajectory(pose_data, frames)
        return frames
    return None
