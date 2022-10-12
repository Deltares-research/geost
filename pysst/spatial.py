import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from typing import List, Union


def header_to_geopandas(entries_df) -> gpd.GeoDataFrame:
    points = [Point([x, y]) for x, y in zip(entries_df.x, entries_df.y)]
    return gpd.GeoDataFrame(entries_df, geometry=points)


def header_from_points(entries_df, point_gdf, buffer, invert) -> gpd.GeoDataFrame:
    data_points = entries_df[["x", "y"]].values
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
    return entries_df[bool_array]


def header_from_lines(header_df, line_gdf, buffer, invert) -> gpd.GeoDataFrame:
    return gpd.sjoin(header_df, line_gdf)[["nr", "x", "y", "mv", "end", "geometry"]]


def header_from_polygons(header_df, polygon_gdf, buffer, invert) -> gpd.GeoDataFrame:
    return gpd.sjoin(header_df, polygon_gdf)[["nr", "x", "y", "mv", "end", "geometry"]]


def find_area_labels(header_df, polygon_gdf, column_name):
    all_nrs = header_df["nr"]
    joined = gpd.sjoin(header_df, polygon_gdf)[column_name]
    # Remove any duplicated indices, which may sometimes happen
    joined = joined[~joined.index.duplicated()]
    return pd.concat([all_nrs, joined], axis=1)
