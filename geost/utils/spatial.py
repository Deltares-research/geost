import warnings
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from geost.utils import conversion


def check_and_coerce_crs(gdf: gpd.GeoDataFrame, to_crs: int):
    """
    Check the CRS of a geodataframe against given crs.

    If the geodataframe has no CRS, makes the geodataframe assume the given CRS. The
    user is warned when this occurs.

    If the geodataframe has a different known CRS, inform user and coerce the CRS
    to the given CRS (to_crs argument).

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe to be converted to the given CRS.
    to_crs : int
        EPSG number of the CRS to check and coerce to.

    Returns
    -------
    GeoDataFrame
        Geodataframe coerced to have the desired CRS
    """
    if gdf.crs is None:
        gdf.crs = to_crs
        warnings.warn(
            "The selection geometry has no crs! Assuming it is the same as the "
            + f"horizontal_reference (epsg:{to_crs}) of this "
            + "collection. PLEASE CHECK WHETHER THIS IS CORRECT!",
            UserWarning,
        )
    elif gdf.crs != to_crs:
        gdf = gdf.to_crs(to_crs)
        warnings.warn(
            "The crs of the selection geometry does not match the horizontal "
            + "reference of the collection. The selection geometry was coerced "
            + f"to epsg:{to_crs} automatically",
            UserWarning,
        )
    return gdf


def select_points_within_bbox(
    gdf: str | Path | gpd.GeoDataFrame,
    xmin: float | int,
    ymin: float | int,
    xmax: float | int,
    ymax: float | int,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on a user-given bounding box.

    Parameters
    ----------
    gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    xmin : float | int
        Minimum x-coordinate of the bounding box.
    ymin : float | int
        Minimum y-coordinate of the bounding box.
    xmax : float | int
        Maximum x-coordinate of the bounding box.
    ymax : float | int
        Maximum y-coordinate of the bounding box.
    invert : bool, optional
        Invert the selection, so select all objects outside of the
        bounding box in this case, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        Geodataframe containing only selected geometries.

    """
    # Instance checks and coerce to geodataframe if required
    gdf = conversion.check_geometry_instance(gdf)
    selected = gdf.cx[xmin:xmax, ymin:ymax]

    if invert:
        selected = gdf.loc[~gdf.index.isin(selected.index)]

    return selected


def select_points_near_points(
    gdf: str | Path | gpd.GeoDataFrame,
    point_gdf: str | Path | gpd.GeoDataFrame,
    buffer: float | int,
    n_points: int = None,
    return_pairs: bool = False,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on point geometries and a buffer.

    Parameters
    ----------
    gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    point_gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select with.
    buffer : float | int
        Buffer distance for selection geometries.
    n_points : int, optional
        Number of nearest points to select, by default None, which means that all points
        within the buffer are selected.
    return_pairs : bool, optional
        Return a dataframe with the pairs of selected points, by default False.
    invert : bool, optional
        Invert the selection, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        Geodataframe containing only selected geometries.
    np.ndarray, optional
        Array containing the pairs of selected points, only returned if `return_pairs` is True.
        The array has shape (n_pairs, 2), where each row contains the indices of the selected points
        in the original `gdf` (data points) in position 0 and `point_gdf` (query points) in position 1.
    """
    from scipy.spatial import KDTree

    # Instance checks and coerce to geodataframe if required
    gdf = conversion.check_geometry_instance(gdf)
    point_gdf = conversion.check_geometry_instance(point_gdf)

    # Selection logic
    data_points = np.array([gdf["geometry"].x, gdf["geometry"].y]).transpose()
    query_points = np.array(
        [point_gdf["geometry"].x, point_gdf["geometry"].y]
    ).transpose()

    data_tree = KDTree(data_points)

    if n_points is None:
        index = data_tree.query_ball_point(query_points, buffer, workers=-1)
        selection_index = np.array([i for sublist in index for i in sublist])

    elif isinstance(n_points, int):
        distances, index = data_tree.query(
            query_points, k=n_points, distance_upper_bound=buffer, workers=-1
        )
        selection_index = index[np.isfinite(distances)]

    if return_pairs:
        pairs = np.array(
            [
                (i, j)
                for i, sublist in enumerate(index)
                for j in sublist
                if j != data_tree.n
            ]
        )

    if invert:
        gdf_reindexed = gdf.copy().reset_index(drop=True)
        gdf_selected = gdf.iloc[~gdf_reindexed.index.isin(selection_index)]
    else:
        gdf_selected = gdf.iloc[selection_index]

    if return_pairs:
        return gdf_selected, pairs
    return gdf_selected


def select_points_near_lines(
    gdf: str | Path | gpd.GeoDataFrame,
    line_gdf: str | Path | gpd.GeoDataFrame,
    buffer: float | int,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on line geometries and a buffer.

    Parameters
    ----------
    gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    line_gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select with.
    buffer : float | int
        Buffer distance for selection geometries.
    invert : bool, optional
        Invert the selection, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        Geodataframe containing only selected geometries.
    """
    # Instance checks and coerce to geodataframe if required
    gdf = conversion.check_geometry_instance(gdf)
    line_gdf = conversion.check_geometry_instance(line_gdf)

    # Selection logic
    line_gdf["geometry"] = line_gdf.buffer(distance=buffer)
    if invert:
        gdf_selected = gdf[~gdf.geometry.within(line_gdf.union_all())]
    else:
        gdf_selected = gdf[gdf.geometry.within(line_gdf.union_all())]
    return gdf_selected


def select_points_within_polygons(
    gdf: str | Path | gpd.GeoDataFrame,
    polygon_gdf: str | Path | gpd.GeoDataFrame,
    buffer: float | int = 0,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on polygon geometries and an optional
    buffer.

    Parameters
    ----------
    gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    polygon_gdf : str | Path | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select with.
    buffer : float | int, optional
        Optional buffer distance around the polygon selection geometries, by default 0.
    invert : bool, optional
        Invert the selection, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        Geodataframe containing only selected geometries.

    """
    # Instance checks and coerce to geodataframe if required
    gdf = conversion.check_geometry_instance(gdf)
    polygon_gdf = conversion.check_geometry_instance(polygon_gdf)

    # Selection logic
    if buffer > 0:
        polygon_select = polygon_gdf.copy()
        polygon_select["geometry"] = polygon_gdf.geometry.buffer(buffer)
    else:
        polygon_select = polygon_gdf

    if invert:
        gdf_selected = gdf[~gdf.geometry.within(polygon_select.union_all())]
    else:
        gdf_selected = gdf[gdf.geometry.within(polygon_select.union_all())]

    return gdf_selected


def get_raster_values(
    x: np.ndarray, y: np.ndarray, raster_to_read: str | Path | xr.DataArray
) -> np.ndarray:
    """
    Return sampled values from a raster at the given (x, y) locations.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates, same length as 'y'.
    y : np.ndarray
        1D array of y-coordinates, same length as 'x'.
    raster_to_read : str | Path | xr.DataArray
        Location of a raster file or an xr.DataArray with dimensions 'x' and 'y'. This
        raster is used to sample values from at all (x, y) locations.

    Returns
    -------
    np.ndarray
        1D array of sampled values
    """
    if isinstance(raster_to_read, (str, Path)):
        raster_to_read = rioxarray.open_rasterio(raster_to_read).squeeze()

    if set(raster_to_read.dims) != set(("y", "x")):
        raise TypeError(
            "The xr.DataArray to sample from does not have the "
            + "required 'x' and 'y' dimensions"
        )

    xmin, ymin, xmax, ymax = raster_to_read.rio.bounds()
    outside_x = (x < xmin) | (x > xmax)
    outside_y = (y < ymin) | (y > ymax)

    surface_levels = raster_to_read.sel(
        x=xr.DataArray(x, dims=("loc")),
        y=xr.DataArray(y, dims=("loc")),
        method="nearest",
    ).values

    surface_levels[outside_x | outside_y] = np.nan

    return surface_levels
