"""Visualize GPS trajectory.

To install the gmplot package, call:

    pip install gmplot

To create a Google Map API key, please follow the instruction below:

    https://developers.google.com/maps/documentation/embed/get-api-key
"""

import argparse
import json
import os
from glob import glob

import gmplot
import numpy as np
from scalabel.common.typing import NDArrayF64

from ..common.logger import logger


def visualize_file(in_file: str, out_file: str, apikey: str) -> None:
    """Visualize one GPS file.

    Given a input json file with gps trajectory and the Google Map API Key,
    creates an html that displays the given trajectory.

    Arguments:
        in_file:  source json file
        out_file: path to save output html
        apikey:   Google Map API key
    """
    # open info json
    with open(in_file, "r", encoding="utf-8") as f:
        info = json.loads(f.read())

    latitude_list = []
    longitude_list = []

    if "gps" not in info:
        logger.info("Field 'gps' not found.")
        return
    if len(info["gps"]) < 1:
        logger.info("Empty trajectory data.")
        return

    for location in info["gps"]:
        latitude_list.append(location["latitude"])
        longitude_list.append(location["longitude"])

    latitude_array: NDArrayF64 = np.array(latitude_list)
    longitude_array: NDArrayF64 = np.array(longitude_list)

    mean_latitude = latitude_array.mean()
    mean_longitude = longitude_array.mean()

    gmap3 = gmplot.GoogleMapPlotter(
        mean_latitude, mean_longitude, 18, apikey=apikey
    )

    # scatter method of map object
    # scatter points on the google map
    gmap3.scatter(
        latitude_array, longitude_array, "# FF0000", size=1, marker=False
    )

    # Plot method Draw a line in
    # between given coordinates
    gmap3.plot(
        latitude_array, longitude_array, "cornflowerblue", edge_width=2.5
    )

    gmap3.draw(out_file)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="source directory or file, if a directory is "
        "given, the script will produce a visualization "
        "for each of the json file found inside the "
        "directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory for generated maps",
    )
    parser.add_argument("-k", "--apikey", default=None)

    args = parser.parse_args()

    assert args.input is not None

    # Check if a directory is given
    args.isdir = os.path.isdir(args.input)

    # if destination is not given, save html files to the same directory as
    # source json files
    if args.output is None:
        if args.isdir:
            args.output = args.input
        else:
            args.output = args.input.replace("json", "html")
    # Otherwise save to the provided destination
    else:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        assert os.path.isdir(args.output)

    return args


def main() -> None:
    """Main."""
    args = parse_args()

    if args.isdir:
        file_list = glob(f"{args.input}/*.json")

        for in_file in file_list:
            out_file = (
                f"{args.output}/"
                f"{in_file.split('/')[-1].replace('.json', '')}.html"
            )
            visualize_file(in_file, out_file, args.apikey)
    else:
        out_file = (
            f"{args.output}/"
            f"{args.input.split('/')[-1].replace('.json', '')}.html"
        )
        visualize_file(args.input, out_file, args.apikey)


if __name__ == "__main__":
    main()
