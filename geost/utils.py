import operator
from pathlib import Path, WindowsPath
from typing import Union

import geopandas as gpd
import pandas as pd
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
    file: Union[str, WindowsPath], out_file: Union[str, WindowsPath] = None, **kwargs
) -> None:
    """
    Convert csv table to parquet.

    Parameters
    ----------
    file : Union[str, WindowsPath]
        Path to csv file to convert.
    out_file : Union[str, WindowsPath], optional
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
    file: Union[str, WindowsPath], out_file: Union[str, WindowsPath] = None, **kwargs
) -> None:
    """
    Convert excel table to parquet.

    Parameters
    ----------
    file : Union[str, WindowsPath]
        Path to excel file to convert.
    out_file : Union[str, WindowsPath], optional
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


def get_path_iterable(path: WindowsPath, wildcard: str = "*"):
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


def warn_user(func):
    def inner(*args, **kwargs):
        print("WARNING:\n--------")
        func(*args, **kwargs)
        print("--------\n>> CONTINUING MAY LEAD TO UNEXPECTED RESULTS\n")

    return inner


def inform_user(func):
    def inner(*args, **kwargs):
        print("PLEASE NOTICE:\n--------")
        func(*args, **kwargs)
        print("--------\n")

    return inner


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
