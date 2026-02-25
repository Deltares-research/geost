from pathlib import Path

import geopandas as gpd
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
