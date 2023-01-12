import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from typing import List, Union


def header_to_geopandas(entries_df) -> gpd.GeoDataFrame:
    points = [
        Point([x, y]) for x, y in zip(entries_df.x, entries_df.y)
    ]  # TODO check with shapely 2.0
    header_as_gdf = gpd.GeoDataFrame(entries_df, geometry=points)
    return header_as_gdf


def header_from_bbox(header_df, xmin, xmax, ymin, ymax, invert):
    header_selected = header_df[
        (header_df.x >= xmin)
        & (header_df.x <= xmax)
        & (header_df.y >= ymin)
        & (header_df.y <= ymax)
    ]
    return header_selected


def header_from_points(header_df, point_gdf, buffer, invert) -> gpd.GeoDataFrame:
    data_points = header_df[["x", "y"]].values
    if isinstance(point_gdf, gpd.GeoDataFrame):
        query_points = np.array([list(pnt.coords)[0] for pnt in point_gdf.geometry])
    else:
        query_points = np.array(point_gdf.geometry.coords)

    bool_array = np.full(len(data_points), False)
    for query_point in query_points:
        distance = np.sqrt(
            (query_point[0] - data_points[:, 0]) ** 2
            + (query_point[1] - data_points[:, 1]) ** 2
        )
        bool_array += distance < buffer
    if invert:
        bool_array = np.invert(bool_array)
    header_selected = header_df[bool_array]
    return header_selected


def header_from_lines(header_df, line_gdf, buffer, invert) -> gpd.GeoDataFrame:
    line_gdf["geometry"] = line_gdf.buffer(distance=buffer)
    header_selected = gpd.sjoin(header_df, line_gdf)[
        ["nr", "x", "y", "mv", "end", "geometry"]
    ]
    return header_selected


def header_from_polygons(header_df, polygon_gdf, buffer, invert) -> gpd.GeoDataFrame:
    header_selected = gpd.sjoin(header_df, polygon_gdf)[
        ["nr", "x", "y", "mv", "end", "geometry"]
    ]
    return header_selected


def find_area_labels(header_df, polygon_gdf, column_name):
    all_nrs = header_df["nr"]
    joined = gpd.sjoin(header_df, polygon_gdf)[column_name]
    # Remove any duplicated indices, which may sometimes happen
    joined = joined[~joined.index.duplicated()]
    area_labels = pd.concat([all_nrs, joined], axis=1)
    return area_labels
