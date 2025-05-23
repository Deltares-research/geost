import operator
import sqlite3
from pathlib import Path
from typing import Any, Union

import geopandas as gpd
import pandas as pd
from pyogrio.errors import FieldError
from shapely.geometry import Point

COMPARISON_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}

ARITHMIC_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


def csv_to_parquet(
    file: Union[str, Path], out_file: Union[str, Path] = None, **kwargs
) -> None:
    """
    Convert csv table to parquet.

    Parameters
    ----------
    file : Union[str, Path]
        Path to csv file to convert.
    out_file : Union[str, Path], optional
        Path to parquet file to be written. If not provided it will use the path of
        'file'.
    **kwargs
        pandas.read_csv kwargs. See pandas.read_csv documentation.

    Raises
    ------
    TypeError
        If 'file' is not a csv file
    """
    file = Path(file)
    if file.suffix != ".csv":
        raise TypeError(f"File must be a csv file, but a {file.suffix}-file was given")

    df = pd.read_csv(file, **kwargs)
    if out_file is None:
        df.to_parquet(file.parent / (file.stem + ".parquet"))
    else:
        df.to_parquet(out_file)


def excel_to_parquet(
    file: Union[str, Path], out_file: Union[str, Path] = None, **kwargs
) -> None:
    """
    Convert excel table to parquet.

    Parameters
    ----------
    file : Union[str, Path]
        Path to excel file to convert.
    out_file : Union[str, Path], optional
        Path to parquet file to be written. If not provided it will use the path of
        'file'.
    **kwargs
        pandas.read_excel kwargs. See pandas.read_excel documentation.

    Raises
    ------
    TypeError
        If 'file' is not an xlsx or xls file.
    """
    file = Path(file)
    if file.suffix not in [".xlsx", ".xls"]:
        raise TypeError(
            f"File must be an excel file, but a {file.suffix}-file was given"
        )

    df = pd.read_excel(file, **kwargs)
    if out_file is None:
        df.to_parquet(file.parent / (file.stem + ".parquet"))
    else:
        df.to_parquet(out_file)


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
    points = [Point([x, y]) for x, y in zip(df[x_col_label], df[y_col_label])]
    gdf = gpd.GeoDataFrame(df, geometry=points, crs=crs)
    return gdf


def save_pickle(data: Any, path: str | Path, **kwargs) -> None:
    """
    Save object as a pickle file (.pkl, .pickle) using Pandas to_pickle.

    Parameters
    ----------
    data : Any
        Object to save.
    path : str | Path
        Path to pickle file.
    **kwargs
            pd.to_pickle kwargs. See relevant Pandas documentation.

    """
    pd.to_pickle(data, path, **kwargs)


def _to_geopackage(
    data: gpd.GeoDataFrame, outfile: str | Path, error_note: str, **kwargs
):
    """
    Helper to add GeoST specific information if a Pyogrio error is raised when data is
    exported to Geopackage but contians invalid column names.

    Raises
    ------
    e
        Pyogrio error with added GeoST information when data has invalid column names.
    """
    try:
        data.to_file(outfile, **kwargs)
    except FieldError as e:
        e.add_note(f"Invalid column name in {error_note}, cannot write GPKG.")
        raise e


def create_connection(database: str | Path):
    """
    Create a database connection to a SQLite database.

    Parameters
    ----------
    database: string
        Path/url/etc. to the database to create the connection to.

    Returns
    -------
    conn : sqlite3.Connection
        Connection object or None.

    """
    conn = None
    try:
        conn = sqlite3.connect(database)
    except sqlite3.Error as e:
        print(e)

    return conn
