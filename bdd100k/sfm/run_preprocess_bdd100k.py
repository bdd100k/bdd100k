"""Scripts for downloading and preprocessing BDD100k data from server."""
import argparse
import os
import pickle

import numpy as np

from .map import visualize_antpath, visualize_map
from .utils import gps_to_m, load_pose_data

PKL_PATH = "/srv/beegfs02/scratch/bdd100k/data/sfm/pkl"
with open(os.path.join(PKL_PATH, "det_label.pkl"), "rb") as f:
    DET_LABEL = pickle.load(f)


class Sequence:
    """Sequence object specific designed for SfM."""

    def __init__(self, name, image_source_path):
        self.name = name
        self.image_tag = name[:-5] + ".jpg"
        self.image_source_path = image_source_path
        # Path for train, test or val
        self.path = ""
        self.scene = ""
        self.weather = ""
        self.timeofday = ""

    def update_info(self):
        """Update sequence by checking the info file."""
        for path in os.listdir(self.image_source_path):
            if os.path.exists(
                os.path.join(self.image_source_path, path, self.image_tag)
            ):
                self.path = path
                break
        if self.path != "":
            if self.image_tag in DET_LABEL[self.path]:
                if "attributes" in DET_LABEL[self.path][self.image_tag]:
                    self.timeofday = DET_LABEL[self.path][self.image_tag][
                        "attributes"
                    ]["timeofday"]
                    self.scene = DET_LABEL[self.path][self.image_tag][
                        "attributes"
                    ]["scene"]
                    self.weather = DET_LABEL[self.path][self.image_tag][
                        "attributes"
                    ]["weather"]


def get_lat_lon(args, seq, seqs_dict):
    """
    Extract latitude and longitude info from the info.json.
    If the sequnce is not yet hashed in the dictionary, add it
    """
    if seq in seqs_dict.keys():
        lat_ = np.expand_dims(seqs_dict[seq]["lat"], 0)
        lon_ = np.expand_dims(seqs_dict[seq]["lon"], 0)
        return lat_, lon_, seqs_dict
    if seq[-5:] != ".json":
        seq += ".json"
    info_path = os.path.join(args.info_folder, seq)
    pose_data = load_pose_data(info_path)
    lat_ = np.array([d["latitude"] for d in pose_data])
    lon_ = np.array([d["longitude"] for d in pose_data])
    seqs_dict[seq] = {}
    seqs_dict[seq]["lat"] = lat_
    seqs_dict[seq]["lon"] = lon_
    lat_ = np.expand_dims(lat_, 0)
    lon_ = np.expand_dims(lon_, 0)
    return lat_, lon_, seqs_dict


def get_overlaps(args, seqs):
    """Given a list of sequences, search and combine overlapped sequences."""
    overlaps = {"unique_seq": {}}
    seqs_dict = {}

    for i, seq_obj in enumerate(seqs[:-1]):
        seq = seq_obj.name
        lat_cur, lon_cur, seqs_dict = get_lat_lon(args, seq, seqs_dict)

        for seq_obj_to_compare in seqs[i + 1 :]:
            seq_to_compare = seq_obj_to_compare.name
            if (
                seq_obj.weather != seq_obj_to_compare.weather
                or seq_obj.scene != seq_obj_to_compare.scene
            ):
                continue

            lat_to_compare, lon_to_compare, seqs_dict = get_lat_lon(
                args, seq_to_compare, seqs_dict
            )
            if len(lat_to_compare[0]) < args.min_gps_frame:
                continue
            lat_comparison = np.absolute(lat_cur - lat_to_compare.T)
            lon_comparison = np.absolute(lon_cur - lon_to_compare.T)
            # 111139 is the approximated conversion from degree to meter
            dist_comparison = 111139 * np.sqrt(
                lat_comparison ** 2 + lon_comparison ** 2
            )
            if np.min(dist_comparison) < float(args.max_gps_distance):
                overlaped_seqs = overlaps["unique_seq"].keys()
                id_overlaps = len(overlaps) - 1
                if (
                    seq not in overlaped_seqs
                    and seq_to_compare not in overlaped_seqs
                ):
                    overlaps[id_overlaps] = [seq, seq_to_compare]
                    overlaps["unique_seq"][seq] = id_overlaps
                    overlaps["unique_seq"][seq_to_compare] = id_overlaps

                elif (
                    seq in overlaped_seqs
                    and seq_to_compare not in overlaped_seqs
                ):
                    id_to_add = overlaps["unique_seq"][seq]
                    overlaps["unique_seq"][seq_to_compare] = id_to_add
                    overlaps[id_to_add].append(seq_to_compare)

                elif (
                    seq not in overlaped_seqs
                    and seq_to_compare in overlaped_seqs
                ):
                    id_to_add = overlaps["unique_seq"][seq_to_compare]
                    overlaps["unique_seq"][seq] = id_to_add
                    overlaps[id_to_add].append(seq)
    return overlaps


def get_singles(seqs_timeofday):
    """Given a dict of sequences, search single sequences."""
    all_seqs = [seq.name for seq in seqs_timeofday["seqs"]]
    overlaps = seqs_timeofday["overlaps"]["unique_seq"]
    singles = list(set(all_seqs) - set(overlaps))
    return singles


def find_path_type(image_tag, image_source_path):
    """Find if sequence belongs to test, train or val"""
    for path in os.listdir(image_source_path):
        if os.path.exists(os.path.join(image_source_path, path, image_tag)):
            return path
    return None


def summarize_postcode(args, target_path, output_path):
    """
    Summarize all overlaps and singles in postcode and save result
    """
    with open(os.path.join(PKL_PATH, "postcode_info_100k.pkl"), "rb") as fid:
        postcode_info = pickle.load(fid)
    key = args.postcode
    all_seqs = postcode_info[key]
    postcode_path = os.path.join(target_path, key)
    os.makedirs(postcode_path, exist_ok=True)

    seqs_timesofday = {
        "daytime": {
            "seqs": [],
            "static_seqs": [],
            "overlaps": {},
            "singles": {},
        },
        "night": {
            "seqs": [],
            "static_seqs": [],
            "overlaps": {},
            "singles": {},
        },
        "dawn_dusk": {
            "seqs": [],
            "static_seqs": [],
            "overlaps": {},
            "singles": {},
        },
    }

    for seq in all_seqs:
        seq_obj = Sequence(seq, args.image_source_path)
        seq_obj.update_info()
        cur_info_path = os.path.join(args.info_folder, seq)
        pose_data = load_pose_data(cur_info_path)
        dist_moved = gps_to_m(
            pose_data[0]["latitude"],
            pose_data[0]["longitude"],
            pose_data[-1]["latitude"],
            pose_data[-1]["longitude"],
        )
        if seq_obj.timeofday in ["daytime", "night", "dawn_dusk"]:
            # Only consider a sequence if
            # enough distance movement detected and
            # enough gps recording
            if (
                len(pose_data) < args.min_gps_frame
                or dist_moved < args.min_dist_moved
            ):
                seqs_timesofday[seq_obj.timeofday]["static_seqs"].append(
                    seq_obj
                )
            else:
                seqs_timesofday[seq_obj.timeofday]["seqs"].append(seq_obj)

    for timeofday in ["daytime", "night", "dawn_dusk"]:
        seqs_timeofday = seqs_timesofday[timeofday]
        timeofday_path = os.path.join(postcode_path, timeofday)
        os.makedirs(timeofday_path, exist_ok=True)
        overlaps_path = os.path.join(timeofday_path, "overlaps")
        os.makedirs(overlaps_path, exist_ok=True)
        singles_path = os.path.join(timeofday_path, "singles")
        os.makedirs(singles_path, exist_ok=True)

        seqs_timeofday["overlaps"] = get_overlaps(args, seqs_timeofday["seqs"])
        seqs_map_name = f"{key}_overlaps.html"
        seqs_list = []
        if len(seqs_timeofday["overlaps"]) > 1:
            for i in range(len(seqs_timeofday["overlaps"]) - 1):
                for seq in seqs_timeofday["overlaps"][i]:
                    seq_info = os.path.join(args.info_folder, seq)
                    seqs_list.append(load_pose_data(seq_info))
            visualize_map(
                seqs_list, os.path.join(timeofday_path, seqs_map_name)
            )

        seqs_timeofday["singles"] = get_singles(seqs_timeofday)
        seqs_map_name = f"{key}_singles.html"
        seqs_list = []
        if len(seqs_timeofday["singles"]) > 0:
            for seq in seqs_timeofday["singles"]:
                seq_info = os.path.join(args.info_folder, seq)
                seqs_list.append(load_pose_data(seq_info))
            visualize_map(
                seqs_list, os.path.join(timeofday_path, seqs_map_name)
            )

    with open(output_path, "wb") as fid:
        pickle.dump(seqs_timesofday, fid)


def main():
    """
    For a postcode and a timeofday, prepare bdd100k dataset for SfM and MVS.
    """
    parser = argparse.ArgumentParser(description="Prepare bdd100k dataset")
    parser.add_argument(
        "--postcode",
        help="Which postcode to run",
    )
    parser.add_argument(
        "--timeofday",
        default="daytime",
        help="Which timeofday to process(daytime, night, dawn_dusk)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If you want to visualize",
    )
    # Must activated if you process the postcode for the first time
    parser.add_argument(
        "--summarize-postcode",
        action="store_true",
        help="If you want to summarize first",
    )
    parser.add_argument(
        "--save-visualize",
        action="store_true",
        help="If you want to save visualization",
    )
    parser.add_argument(
        "--min-gps-frame",
        default=30,
        help="Number of frame of gps for a sequence to be considered",
    )
    parser.add_argument(
        "--min-dist-moved",
        default=10,
        help="Minimum distance moved for a sequence to be considered",
    )
    parser.add_argument(
        "--max-gps-distance",
        default=12,
        help="Max distance between 2 gps coords to be considered as overlap",
    )
    parser.add_argument(
        "--info-folder",
        default="/srv/beegfs02/scratch/bdd100k/data/bdd100k/info",
        help="path to all infos",
    )
    parser.add_argument(
        "--target-path",
        default="/scratch_net/zoidberg_second/yuthan/bdd100k",
        help="path to save all data",
    )
    parser.add_argument(
        "--image-source-path",
        default="/srv/beegfs02/scratch/bdd100k/data/bdd100k/images/100k",
        help="path to BDD100K info files",
    )

    args = parser.parse_args()
    key = args.postcode
    target_path = args.target_path
    postcode_path = os.path.join(target_path, key)
    os.makedirs(postcode_path, exist_ok=True)
    postcode_info_pkl = os.path.join(postcode_path, f"{key}_seqs.pkl")
    if args.summarize_postcode:
        summarize_postcode(args, target_path, postcode_info_pkl)

    with open(postcode_info_pkl, "rb") as fid:
        seqs_timesofday = pickle.load(fid)
    timeofday = args.timeofday
    seqs_timeofday = seqs_timesofday[timeofday]
    timeofday_path = os.path.join(postcode_path, timeofday)

    # Prepare data for all overlaps
    overlaps_path = os.path.join(timeofday_path, "overlaps")
    os.makedirs(overlaps_path, exist_ok=True)
    for i in range(len(seqs_timeofday["overlaps"]) - 1):
        overlap_path = os.path.join(overlaps_path, str(i))
        if os.path.exists(overlap_path):
            continue
        os.makedirs(overlap_path, exist_ok=True)
        image_tags_path = os.path.join(overlap_path, "image_tags")
        os.makedirs(image_tags_path, exist_ok=True)
        zip_path = os.path.join(overlap_path, "zip")
        os.makedirs(zip_path, exist_ok=True)
        image_path = os.path.join(overlap_path, "images")
        os.makedirs(image_path, exist_ok=True)

        # Copy image tags and plotting overlap
        list_seq = []
        for seq in seqs_timeofday["overlaps"][i]:
            cur_info_path = os.path.join(args.info_folder, seq)
            list_seq.append(load_pose_data(cur_info_path))
            image_tag = seq[:-5] + ".jpg"
            path_type = find_path_type(image_tag, args.image_source_path)
            image_tag_path = os.path.join(
                args.image_source_path, path_type, image_tag
            )
            os.system(f"cp {image_tag_path} {image_tags_path}")
        antpath_name = f"{key}_{i}.html"
        visualize_antpath(list_seq, os.path.join(overlap_path, antpath_name))

        # Save the information about the sequences inside one overlap
        with open(os.path.join(overlap_path, "seq_info.pkl"), "wb") as fid:
            pickle.dump(seqs_timeofday["overlaps"][i], fid)

        # Copy and resample all images
        for image_tags in os.listdir(image_tags_path):
            zip_name = image_tags[:-4] + ".zip"
            os.system(
                f"wget http://dl.yf.io/bdd100k/images/all_zip/{zip_name} "
                f"-P {zip_path}"
            )

        # Unzip zips, generate resampled seq, create image folders
        for file_ in os.listdir(zip_path):
            unzip_path = os.path.join(overlap_path, "unzip", file_[:-4])
            os.makedirs(unzip_path, exist_ok=True)
            os.system(f"unzip {os.path.join(zip_path, file_)} -d {unzip_path}")
            for image_ in os.listdir(unzip_path):
                id_ = int(image_[-8:-4])
                if id_ % 6 == 1:
                    os.system(
                        f"cp {os.path.join(unzip_path, image_)} {image_path}"
                    )
            os.system(f"rm -r {unzip_path}")
        os.system(f"rm -r {zip_path}")
        empty_unzip_pah = os.path.join(overlap_path, "unzip")
        os.system(f"rm -r {empty_unzip_pah}")

    # Prepare data for all single sequences
    singles_path = os.path.join(timeofday_path, "singles")
    os.makedirs(singles_path, exist_ok=True)
    for seq in seqs_timeofday["singles"]:
        single_path = os.path.join(
            singles_path, seq[:-5]
        )  # singles path contain all single path
        if os.path.exists(single_path):
            continue
        os.makedirs(single_path, exist_ok=True)
        image_tags_path = os.path.join(single_path, "image_tags")
        os.makedirs(image_tags_path, exist_ok=True)
        zip_path = os.path.join(single_path, "zip")
        os.makedirs(zip_path, exist_ok=True)
        image_path = os.path.join(single_path, "images")
        os.makedirs(image_path, exist_ok=True)

        image_tag = seq[:-5] + ".jpg"
        path_type = find_path_type(image_tag, args.image_source_path)
        os.system(
            f"cp {os.path.join(args.image_source_path, path_type, image_tag)} "
            f"{image_tags_path}"
        )
        cur_info_path = os.path.join(args.info_folder, seq)
        list_seq = [load_pose_data(cur_info_path)]
        antpath_name = f"{key}_{seq[:-5]}.html"
        visualize_antpath(list_seq, os.path.join(single_path, antpath_name))
        for image_tags in os.listdir(image_tags_path):
            zip_name = image_tags[:-4] + ".zip"
            os.system(
                f"wget http://dl.yf.io/bdd100k/images/all_zip/{zip_name} "
                f"-P {zip_path}"
            )

        # Unzip zips, generate resampled seq, create image folders
        for file_ in os.listdir(zip_path):
            unzip_path = os.path.join(single_path, "unzip", file_[:-4])
            os.makedirs(unzip_path, exist_ok=True)
            os.system(f"unzip {os.path.join(zip_path, file_)} -d {unzip_path}")
            for image_ in os.listdir(unzip_path):
                id_ = int(image_[-8:-4])
                if id_ % 6 == 1:
                    os.system(
                        f"cp {os.path.join(unzip_path, image_)} {image_path}"
                    )
            os.system(f"rm -r {unzip_path}")
        os.system(f"rm -r {zip_path}")
        empty_unzip_pah = os.path.join(single_path, "unzip")
        os.system(f"rm -r {empty_unzip_pah}")


if __name__ == "__main__":
    main()
