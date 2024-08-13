from pathlib import Path, WindowsPath

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from geost.utils import inform_user, warn_user

warn = warn_user(lambda warning_info: print(warning_info))
inform = inform_user(lambda info: print(info))


def check_gdf_instance(
    gdf_or_path: str | WindowsPath | gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Check if the argument is already a geodataframe or pointing to a file
    containing geometries

    Parameters
    ----------
    gdf_or_path : str | WindowsPath | gpd.GeoDataFrame
        The geodataframe or file that can be read and parsed to a geodataframe

    Returns
    -------
    gpd.GeoDataFrame
        An instance of a geopandas geodataframe
    """
    if isinstance(gdf_or_path, str | WindowsPath):
        gdf_or_path = Path(gdf_or_path)
        filetype = gdf_or_path.suffix
        if filetype in (".parquet", ".geoparquet"):
            gdf = gpd.read_parquet(gdf_or_path)
        else:
            gdf = gpd.read_file(gdf_or_path)
    elif isinstance(gdf_or_path, gpd.GeoDataFrame):
        gdf = gdf_or_path

    # Make sure you don't get a view of gdf returned
    gdf = gdf.copy()

    return gdf


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
        warn(
            "The selection geometry has no crs! Assuming it is the same as the "
            + f"horizontal_reference (epsg:{to_crs}) of this "
            + "collection. PLEASE CHECK WHETHER THIS IS CORRECT!",
        )
    elif gdf.crs != to_crs:
        gdf = gdf.to_crs(to_crs)
        inform(
            "The crs of the selection geometry does not match the horizontal "
            + "reference of the collection. The selection geometry was coerced "
            + f"to epsg:{to_crs} automatically"
        )
    return gdf


def select_points_within_bbox(
    gdf: str | WindowsPath | gpd.GeoDataFrame,
    xmin: float | int,
    xmax: float | int,
    ymin: float | int,
    ymax: float | int,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on a user-given bounding box.

    Parameters
    ----------
    gdf : str | WindowsPath | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    xmin : float | int
        Minimum x-coordinate of the bounding box.
    xmax : float | int
        Maximum x-coordinate of the bounding box.
    ymin : float | int
        Minimum y-coordinate of the bounding box.
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
    gdf = check_gdf_instance(gdf)

    # Selection logic
    x = gdf["geometry"].x
    y = gdf["geometry"].y
    if invert:
        gdf_selected = gdf[(x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)]
    else:
        gdf_selected = gdf[(x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)]
    return gdf_selected


def select_points_near_points(
    gdf: str | WindowsPath | gpd.GeoDataFrame,
    point_gdf: str | WindowsPath | gpd.GeoDataFrame,
    buffer: float | int,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on point geometries and a buffer.

    Parameters
    ----------
    gdf : str | WindowsPath | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    point_gdf : str | WindowsPath | gpd.GeoDataFrame
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
    gdf = check_gdf_instance(gdf)
    point_gdf = check_gdf_instance(point_gdf)

    # Selection logic
    data_points = np.array([gdf["geometry"].x, gdf["geometry"].y]).transpose()
    query_points = np.array(
        [point_gdf["geometry"].x, point_gdf["geometry"].y]
    ).transpose()

    bool_array = np.full(len(data_points), False)
    for query_point in query_points:
        distance = np.sqrt(
            (query_point[0] - data_points[:, 0]) ** 2
            + (query_point[1] - data_points[:, 1]) ** 2
        )
        bool_array += distance < buffer
    if invert:
        bool_array = np.invert(bool_array)
    gdf_selected = gdf[bool_array]
    return gdf_selected


def select_points_near_lines(
    gdf: str | WindowsPath | gpd.GeoDataFrame,
    line_gdf: str | WindowsPath | gpd.GeoDataFrame,
    buffer: float | int,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on line geometries and a buffer.

    Parameters
    ----------
    gdf : str | WindowsPath | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    line_gdf : str | WindowsPath | gpd.GeoDataFrame
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
    gdf = check_gdf_instance(gdf)
    line_gdf = check_gdf_instance(line_gdf)

    # Selection logic
    line_gdf["geometry"] = line_gdf.buffer(distance=buffer)
    if invert:
        gdf_selected = gdf[~gdf.geometry.within(line_gdf.union_all())]
    else:
        gdf_selected = gdf[gdf.geometry.within(line_gdf.union_all())]
    return gdf_selected


def select_points_within_polygons(
    gdf: str | WindowsPath | gpd.GeoDataFrame,
    polygon_gdf: str | WindowsPath | gpd.GeoDataFrame,
    buffer: float | int = 0,
    invert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Make a selection of point geometries based on polygon geometries and an optional
    buffer.

    Parameters
    ----------
    gdf : str | WindowsPath | gpd.GeoDataFrame
        Geodataframe (or file that can be parsed to a geodataframe) to select from.
    polygon_gdf : str | WindowsPath | gpd.GeoDataFrame
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
    gdf = check_gdf_instance(gdf)
    polygon_gdf = check_gdf_instance(polygon_gdf)

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


def find_area_labels(
    point_geodataframe: gpd.GeoDataFrame,
    polygon_geodataframe: gpd.GeoDataFrame,
    column_name: str,
) -> pd.Series:
    """
    Function to find labels associated with polygon geometries for a series of queried
    point geometries. Basically a spatial join between the point and polygon dataframe.

    Parameters
    ----------
    point_geodataframe : gpd.GeoDataFrame
        Geodataframe with point geometries for which you want to find in which polygon
        geometries they are located.
    polygon_geodataframe : gpd.GeoDataFrame
        Geodataframe with polygon geometries
    column_name : str
        Label of the polygon geometries to use for assigning to the queried points

    Returns
    -------
    pandas.Series
        Series with labels from the polygon geometries for each point.
    """
    joined = gpd.sjoin(point_geodataframe, polygon_geodataframe)[column_name]
    # Remove any duplicated indices, which may sometimes happen
    area_labels = joined[~joined.index.duplicated()]
    return area_labels


def get_raster_values(
    x: np.ndarray, y: np.ndarray, raster_to_read: str | WindowsPath | xr.DataArray
) -> np.ndarray:
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
        raster_to_read = rioxarray.open_rasterio(raster_to_read).squeeze()

    if set(raster_to_read.dims) != set(("y", "x")):
        raise TypeError(
            "The xr.DataArray to sample from does not have the "
            + "required 'x' and 'y' dimensions"
        )

    # Determine x/y min/max, accounting for xarray coordinates that represent cell
    # midpoints. Subsequently determine whether the queried x/y values fall outside of
    # the raster_to_read area.
    half_cellsize = np.abs(raster_to_read["x"][1] - raster_to_read["x"][0]).values / 2
    xmin, xmax = (
        raster_to_read["x"].min().values - half_cellsize,
        raster_to_read["x"].max().values + half_cellsize,
    )
    ymin, ymax = (
        raster_to_read["y"].min().values - half_cellsize,
        raster_to_read["y"].max().values + half_cellsize,
    )
    outside_x = (x < xmin) | (x > xmax)
    outside_y = (y < ymin) | (y > ymax)

    surface_levels = raster_to_read.sel(
        x=xr.DataArray(x, dims=("loc")),
        y=xr.DataArray(y, dims=("loc")),
        method="nearest",
    ).values

    surface_levels[outside_x | outside_y] = np.nan

    return surface_levels
