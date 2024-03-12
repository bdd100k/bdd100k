"""Street map visualization of BDD100K sequences."""
from collections import defaultdict
from typing import Dict, List

import numpy as np

try:
    import folium
    import osmnx as ox
    from ipyleaflet import AntPath, Map
except ImportError:
    pass
from scalabel.common.parallel import NPROC, pmap
from scalabel.common.typing import DictStrAny

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
        max(np.max(track[:, 0]) for track in tracks) + 0.001,
        min(np.min(track[:, 0]) for track in tracks) - 0.001,
        max(np.max(track[:, 1]) for track in tracks) + 0.001,
        min(np.min(track[:, 1]) for track in tracks) - 0.001,
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
    m = Map(center=coords[int(len(coords) / 2)], max_zoom=22, zoom=18)
    # basemap=basemaps.OpenStreetMap.Mapnik)
    m.layout.width = "900px"
    m.layout.height = "600px"
    for coords in coords_list:
        antpath = AntPath(locations=coords, delay=2000)
        m.add_layer(antpath)
    m.save(save_path)


def split_sequences_by_city(
    sequences: Sequences, nproc: int = NPROC
) -> Dict[str, Sequences]:
    """Split sequences by geographical area."""
    sequences_by_city: Dict[str, Sequences] = defaultdict(list)
    locations = pmap(
        get_location_from_data, [seq[0] for seq in sequences], nproc
    )
    for loc, seq in zip(locations, sequences):
        sequences_by_city[loc.city].append(seq)
    return sequences_by_city
