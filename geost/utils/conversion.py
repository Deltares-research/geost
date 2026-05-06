import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from geost.utils.io_helpers import _geopandas_read


def get_path_iterable(path: Path, wildcard: str = "*"):
    if path.is_file():
        return [path]
    elif path.is_dir():
        return path.glob(wildcard)
    else:
        raise TypeError("Given path is not a file or a folder")


def safe_float(number):
    try:
        return float(number)
    except ValueError:
        return None


def dataframe_to_geodataframe(
    df: pd.DataFrame, x_col_label: str = "x", y_col_label: str = "y", crs: int = None
) -> gpd.GeoDataFrame:
    """
    Take a dataframe with columns that indicate x and y coordinates and use these to
    turn the dataframe into a geopandas GeoDataFrame with a geometry column that
    contains shapely Point geometries.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns for x and y coordinates.
    x_col_label : str
        Label of the x-coordinate column, default x-coordinate column label is 'x'.
    y_col_label : str
        Label of the y-coordinate column, default y-coordinate column label is 'y'.
    crs : int
        EPSG number as integer.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with point geometries in addition to input dataframe data.

    Raises
    ------
    IndexError
        If input dataframe does not have a valid column for 'x' or 'y'.
    """
    from shapely import points

    pts = points(df[x_col_label], df[y_col_label])
    gdf = gpd.GeoDataFrame(df, geometry=pts, crs=crs)
    return gdf


def check_geometry_instance(
    geometry: str | Path | gpd.GeoDataFrame | BaseGeometry | list[BaseGeometry],
) -> gpd.GeoDataFrame:
    """
    Check if the input geometry is a valid type and convert it to a GeoDataFrame if
    necessary.

    Parameters
    ----------
    geometry : str | Path | gpd.GeoDataFrame | Geometry | list[Geometry]
        The geometry to check.

    Returns
    -------
    gpd.GeoDataFrame
        An instance of a geopandas geodataframe
    """
    if isinstance(geometry, str | Path):
        gdf = _geopandas_read(Path(geometry))
    elif isinstance(geometry, gpd.GeoDataFrame):
        gdf = geometry
    elif isinstance(geometry, BaseGeometry):
        gdf = gpd.GeoDataFrame(geometry=[geometry])
    elif isinstance(geometry, list):
        gdf = gpd.GeoDataFrame(geometry=geometry)

    # Make sure you don't get a view of gdf returned
    gdf = gdf.copy()

    return gdf


def adjust_z_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpret data containing elevation or z-coordinates. GeoST data objects require
    that layer top/bottom or discrete data point z-coordinates to be increasing
    downward, starting at at 0. This function detects how elevation is currently defined
    and turns it into the desired format if required.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with layer or discrete data that includes columns referecing layer
        top, bottoms or discrete data z-coordinates.

    Returns
    -------
    pd.DataFrame
        Dataframe with adjusted z-coordinates if required.

    """
    positional_columns = df.gst.positional_columns

    if not df.gst.has_depth_columns:
        return df  # Do nothing, let validation raise warnings

    if df.gst.is_layered:
        df = _adjust_layered(
            df,
            positional_columns["top"],
            positional_columns["depth"],
            positional_columns["surface"],
        )
    else:
        df = _adjust_discrete(
            df,
            positional_columns["nr"],
            positional_columns["depth"],
            positional_columns["surface"],
        )

    return df


def _adjust_layered(
    df: pd.DataFrame, top: str, bottom: str, surface: str
) -> pd.DataFrame:
    first_row_mask = df.gst.first_row_survey

    surface_values = df.loc[first_row_mask, surface]
    top_values = df.loc[first_row_mask, top]

    if np.allclose(surface_values, top_values):
        df[top] = df[surface] - df[top]
        df[bottom] = df[surface] - df[bottom]
    elif np.allclose(top_values, 0):
        df[top] *= np.sign(df[top])
        df[bottom] *= np.sign(df[bottom])
    else:
        from geost._warnings import MixedDepthWarning

        warnings.warn(
            "Data contains a mix of surveys with depth respect to surface and with depth "
            "starting at 0. GeoST methods including depth expect all surveys to have a. "
            "depth starting at 0 and increasing downward. Please adjust your data accordingly.",
            MixedDepthWarning,
        )

    return df


def _adjust_discrete(
    df: pd.DataFrame, nr: str, depth: str, surface: str
) -> pd.DataFrame:
    first_row_mask = df.gst.first_row_survey
    dz = df.groupby(nr)[depth].transform(lambda x: x.diff().mean())

    surface_values = df.loc[first_row_mask, surface]
    depth_values = df.loc[first_row_mask, depth] - dz[first_row_mask]

    if np.allclose(surface_values, depth_values):
        df[depth] = df[surface] - df[depth]
    elif np.allclose(depth_values, 0):
        df[depth] *= np.sign(df[depth])
    else:
        from geost._warnings import MixedDepthWarning

        warnings.warn(
            "Data contains a mix of surveys with depth respect to surface and with depth "
            "starting at 0. GeoST methods including depth expect all surveys to have a. "
            "depth starting at 0 and increasing downward. Please adjust your data accordingly.",
            MixedDepthWarning,
        )
    return df
