import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from typing import List, Union


def header_to_geopandas(entries_df) -> gpd.GeoDataFrame:
    points = [Point([x, y]) for x, y in zip(entries_df.x, entries_df.y)]
    return gpd.GeoDataFrame(entries_df, geometry=points)


def header_from_polygon(entries_df, polygon_file) -> List[str]:
    polygons = gpd.read_file(polygon_file)
    return gpd.sjoin(entries_df, polygons)[["x", "y", "mv", "end", "geometry"]]


def header_from_line(entries_df, line_file) -> List[str]:
    lines = gpd.read_file(line_file)
    return gpd.sjoin(entries_df, lines)[["x", "y", "mv", "end", "geometry"]]
