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
    # surface, top, bot = df.gst._surface, df.gst._top, df.gst._bottom

    # # For layered data:
    # # First top == surface, depth is with respect to e.g. NAP
    # if top is not None:
    #     relative_to_reference = (df[surface] == df[top])[df.gst.first_row_survey].all()
    #     if relative_to_reference:
    #         pass
    #     else:
    #         pass
    # else:
    #     relative_to_reference = False
    # First top == 0, and depth is positive downward
    # First top == 0, and depth is negative downward

    # For discrete data (first depth is never 0 because it indicates the bottom of intervals):
    # Depth is with respect to e.g. NAP
    # Depth is positive downward with respect to surface
    # Depth is negative downward with respect to surface

    top_column_label = [col for col in df.columns if col in {"top", "depth"}][0]
    has_bottom_column = "bottom" in df.columns

    # TODO: think about detection. Only considers first two indices to determine
    # downward decreasing or increasing of z-coordinates.
    first_surface = df["surface"].iloc[0]
    first_top = df[top_column_label].iloc[0]
    second_top = df[top_column_label].iloc[1]

    if first_top == first_surface:
        df[top_column_label] -= df["surface"]
        if has_bottom_column:
            df["bottom"] -= df["surface"]
    if first_top > second_top:
        df[top_column_label] *= -1
        if has_bottom_column:
            df["bottom"] *= -1

    return df
