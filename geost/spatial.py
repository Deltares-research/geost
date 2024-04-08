from pathlib import WindowsPath
from typing import TypeVar

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

# Local imports
from geost.utils import MissingOptionalModule

# Optional imports
try:
    import geopandas as gpd
except ModuleNotFoundError:
    gpd = MissingOptionalModule("geopandas")

try:
    from shapely.geometry import Point
except ModuleNotFoundError:
    Point = MissingOptionalModule("shapely")

GeoDataFrame = TypeVar("GeoDataFrame")


def header_to_geopandas(entries_df, crs) -> GeoDataFrame:
    points = [
        Point([x, y]) for x, y in zip(entries_df.x, entries_df.y)
    ]  # TODO check with shapely 2.0
    header_as_gdf = gpd.GeoDataFrame(entries_df, geometry=points, crs=crs)
    return header_as_gdf


def header_from_bbox(header_df, xmin, xmax, ymin, ymax, invert) -> GeoDataFrame:
    header_selected = header_df[
        (header_df.x >= xmin)
        & (header_df.x <= xmax)
        & (header_df.y >= ymin)
        & (header_df.y <= ymax)
    ]
    return header_selected


def header_from_points(header_df, point_gdf, buffer, invert) -> GeoDataFrame:
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


def header_from_lines(header_df, line_gdf, buffer, invert) -> GeoDataFrame:
    line_gdf["geometry"] = line_gdf.buffer(distance=buffer)
    header_selected = gpd.sjoin(header_df, line_gdf)[
        ["nr", "x", "y", "mv", "end", "geometry"]
    ]
    return header_selected


def header_from_polygons(header_df, polygon_gdf, buffer, invert) -> GeoDataFrame:
    if buffer > 0:
        polygon_select = polygon_gdf.copy()
        polygon_select["geometry"] = polygon_gdf.geometry.buffer(buffer)
    else:
        polygon_select = polygon_gdf

    if invert:
        header_selected = header_df[
            ~header_df.geometry.within(polygon_select.geometry.unary_union)
        ]
    else:
        header_selected = header_df[
            header_df.geometry.within(polygon_select.geometry.unary_union)
        ]

    return header_selected


def find_area_labels(header_df, polygon_gdf, column_name):
    all_nrs = header_df["nr"]
    joined = gpd.sjoin(header_df, polygon_gdf)[column_name]
    # Remove any duplicated indices, which may sometimes happen
    joined = joined[~joined.index.duplicated()]
    area_labels = pd.concat([all_nrs, joined], axis=1)
    return area_labels


def get_raster_values(
    x: np.ndarray, y: np.ndarray, raster_to_read: str | WindowsPath | xr.DataArray
):
    """
    Return sampled values from a raster at the given (x, y) locations.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates, same length as 'y'.
    y : np.ndarray
        1D array of y-coordinates, same length as 'x'.
    raster_to_read : str | WindowsPath | xr.DataArray
        Location of a raster file or an xr.DataArray with dimensions 'x' and 'y'. This
        raster is used to sample values from at all (x, y) locations.

    Returns
    -------
    np.ndarray
        1D array of sampled values
    """
    if isinstance(raster_to_read, (str, WindowsPath)):
        raster_data = rioxarray.open_rasterio(raster_to_read).squeeze()
    elif isinstance(raster_to_read, xr.DataArray):
        if set(raster_to_read.dims) != set(("y", "x")):
            raise TypeError(
                "The xr.DataArray to sample from does not have the "
                + "required 'x' and 'y' dimensions"
            )
    else:
        raise TypeError("Could not intepret the raster_to_read")

    surface_levels = raster_data.sel(
        x=xr.DataArray(x, dims=("loc")),
        y=xr.DataArray(y, dims=("loc")),
        method="nearest",
    ).values

    return surface_levels
