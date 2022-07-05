"""Street map visualization of BDD100K sequences."""
import argparse
from collections import defaultdict
from typing import Dict, List

import folium
import numpy as np
import osmnx as ox
from scalabel.common.parallel import NPROC, pmap
from scalabel.common.typing import DictStrAny, NDArrayF64
from ipyleaflet import Map, AntPath

from .utils import get_location_from_data, latlon_from_data

Sequences = List[List[DictStrAny]]


def visualize_map(list_sequences: Sequences, save_path: str) -> None:
    """Generate a map visualization for a given set of sequences."""
    track_color = "black"
    track_size = 2
    route_size = 2
    route_color = "black"
    route_opacity = 0.7

    tracks = [latlon_from_data(seq) for seq in list_sequences]
    graph = ox.graph_from_bbox(
        max([np.max(track[:, 0]) for track in tracks]) + 0.001,
        min([np.min(track[:, 0]) for track in tracks]) - 0.001,
        max([np.max(track[:, 1]) for track in tracks]) + 0.001,
        min([np.min(track[:, 1]) for track in tracks]) - 0.001,
        simplify=False,
        retain_all=True,
        network_type="drive",
    )
    street_map = ox.plot_graph_folium(
        graph, popup_attribute="name", edge_width=1, edge_color="darkgrey"
    )

    for track in tracks:
        folium.PolyLine(
            track, color=route_color, weight=route_size, opacity=route_opacity
        ).add_to(street_map)
        for loc in track:
            folium.CircleMarker(
                location=[loc[0], loc[1]],
                radius=track_size,
                weight=1,
                color=track_color,
                fill=True,
                fill_opacity=1,
            ).add_to(street_map)
    street_map.save(save_path)


def visualize_antpath(list_sequences: Sequences, save_path: str) -> None:
    """Generate antpath visualization for a given set of sequences."""
    coords_list = []
    for seq in list_sequences:
        lat_ = np.array([d["latitude"] for d in seq])
        lon_ = np.array([d["longitude"] for d in seq])
        coords = [(lat_[i], lon_[i]) for i in range(len(lat_))]
        coords_list.append(coords)
    m = Map(
        center=coords[int(len(coords)/2)],
        max_zoom=22, zoom=18)
        # basemap=basemaps.OpenStreetMap.Mapnik)
    m.layout.width='900px'
    m.layout.height='600px'
    for coords in coords_list:
        antpath = AntPath(locations=coords, delay=2000)
        m.add_layer(antpath)
    m.save(save_path)


def counterclockwise(point1, point2, point3):
    """Determinine if three points are listed in a counterclockwise order."""
    return (point3[1] - point1[1]) * (point2[0] - point1[0]) > (
        point2[1] - point1[1]
    ) * (point3[0] - point1[0])


def intersect(t1p1, t1p2, t2p1, t2p2):
    """Compute if two line segments (t1p1, t1p2) and (t2p1, t2p2) intersect."""
    return counterclockwise(t1p1, t2p1, t2p2) != counterclockwise(
        t1p2, t2p1, t2p2
    ) and counterclockwise(t1p1, t1p2, t2p1) != counterclockwise(
        t1p1, t1p2, t2p2
    )


def tracks_intersect(track1: NDArrayF64, track2: NDArrayF64) -> bool:
    """Compute if two GPS tracks intersect."""
    for t1p1, t1p2 in zip(track1[:-1], track1[1:]):
        for t2p1, t2p2 in zip(track2[:-1], track2[1:]):
            if intersect(t1p1, t1p2, t2p1, t2p2):
                return True
    return False


def find_intersecting_sequences(sequences: Sequences):
    """Find intersecting paths in given sequences."""
    tracks = [latlon_from_data(seq) for seq in sequences]
    intersecting_tracks = np.zeros((len(tracks), len(tracks)))
    for i, key_track in enumerate(tracks):
        for j, ref_track in enumerate(tracks):
            if tracks_intersect(key_track, ref_track):
                intersecting_tracks[i, j] = 1
    return intersecting_tracks


def split_sequences_by_city(
    sequences: Sequences, nproc: int = NPROC
) -> Dict[str, Sequences]:
    """Split sequences by geographical area."""
    sequences_by_city: Dict[str, Sequences] = defaultdict(list)
    locations = pmap(
        get_location_from_data, [seq[0] for seq in sequences], nproc
    )
    for loc, seq in zip(locations, sequences):
        sequences_by_city[loc.city].append(seq)  # type: ignore
    return sequences_by_city


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument(
        "--info-path",
        type=str,
        choices=["train", "test", "val"],
        help="Path to extract map data from.",
    )
    args = parser.parse_args()
