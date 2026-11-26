import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import geopandas as gpd
import pandas as pd
from pyogrio.errors import FieldError

from geost.utils import conversion
from geost.validation.column_names import check_positional_column_presence

if TYPE_CHECKING:
    from geost.base import Collection


# TODO: This one will also be deprecated when we remove geost.read_table
def _pandas_read(file: str | Path, **kwargs) -> pd.DataFrame:
    """
    Read parquet file.

    Parameters
    ----------
    file : Path
        Path to file to be read.
    kwargs : optional
        Kwargs to pass to the read function

    Returns
    -------
    pd.DataFrame
        Dataframe with contents of 'file'.

    Raises
    ------
    TypeError
        if 'file' has no valid suffix.

    """
    file = Path(file)
    suffix = file.suffix
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file, engine="pyarrow", **kwargs)
    elif suffix in {".csv", ".txt", ".tsv"}:
        geost_attrs = _geost_attrs_from_csv(file)
        if geost_attrs is not None:
            data = pd.read_csv(file, skiprows=1, **kwargs)
            data.attrs = geost_attrs
        else:
            data = pd.read_csv(file, **kwargs)
        return data
    elif suffix in {".xls", ".xlsx"}:  # pragma: no cover
        return pd.read_excel(
            file, **kwargs
        )  # No cover: Excel file io fails in windows CI pipeline
    else:
        raise TypeError(
            f"Expected parquet file (with .parquet or .pq suffix) but got {suffix} file"
        )


def _geopandas_read(file: str | Path, **kwargs) -> gpd.GeoDataFrame:
    """
    Read a geospatial file into a GeoDataFrame.

    Parameters
    ----------
    file : str | Path
        Path to the geospatial file to be read.
    kwargs : optional
        Additional keyword arguments to pass to the geopandas read function. See relevant
        documentation for details.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the data from the file.

    Raises
    ------
    ValueError
        If the file type is not supported.

    """
    file = Path(file)
    if file.suffix in {".shp", ".gpkg"}:
        return gpd.read_file(file, **kwargs)
    elif file.suffix in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(file, **kwargs)
    else:
        raise ValueError(f"File type {file.suffix} is not supported by geopandas.")


def _geost_data_to_collection(
    data: pd.DataFrame,
    coordinate_names: tuple[str, str] | None = None,
    include_in_header: str | Iterable[str] | None = None,
    column_mapper: dict | None = None,
) -> Collection:
    if column_mapper:
        data = data.rename(columns=column_mapper)

    check_positional_column_presence(data)
    data = conversion.adjust_z_coordinates(data)

    # Get stored collection attributes if present.
    if geost_attrs := data.attrs.get("geost"):
        data_columns = geost_attrs.get("data_columns")
        include_in_header = geost_attrs.get("header_columns")
        crs = geost_attrs.get("crs")
        vertical_datum = geost_attrs.get("vertical_datum")
        metadata = geost_attrs.get("metadata")
    else:
        crs = None
        vertical_datum = None
        metadata = None

    data = data.gst.to_collection(
        crs=crs,
        vertical_datum=vertical_datum,
        coordinate_names=coordinate_names,
        include_in_header=include_in_header,
        metadata=metadata,
    )

    if geost_attrs:
        data.data = data.data[data_columns]

    return data


def _geost_attrs_from_csv(file: str | Path) -> dict[str, Any]:
    """
    Extract GeoST attributes from a CSV file.

    Parameters
    ----------
    file : str | Path
        Path to the CSV file to read.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the GeoST attributes.
    """
    with open(file, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")

    if first_line.startswith("#GEOSTATTRS#"):
        attrs = json.loads(first_line[12:])
    else:
        return None

    return attrs


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


def _vertical_datum_from_header(header: gpd.GeoDataFrame) -> str | None:
    """
    Extract the vertical datum from the header GeoDataFrame. This assumes that the column
    "vertical_datum" exists in the header and contains the vertical datum information.

    The column "vertical_datum" will be present when reading parsing BRO XML data

    Parameters
    ----------
    header : gpd.GeoDataFrame
        Header GeoDataFrame containing the vertical datum information.

    Returns
    -------
    str | None
        Vertical datum if present and unique for all surveys, otherwise None.
    """
    if "vertical_datum" in header.columns:
        unique_vertdats = set(header["vertical_datum"])
        if len(unique_vertdats) == 1:
            return unique_vertdats.pop()
    return None


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
