import sqlite3

import geopandas as gpd
import pandas as pd
import pytest
from pyogrio.errors import FieldError
from shapely import geometry as gmt

from geost.utils import io_helpers


@pytest.mark.parametrize("extension", [".geoparquet", ".parquet", ".gpkg", ".shp"])
def test_geopandas_read(tmp_path, point_header, extension):
    outfile = tmp_path / f"test{extension}"

    if extension in {".geoparquet", ".parquet"}:
        point_header.to_parquet(outfile)
    elif extension in {".gpkg", ".shp"}:
        point_header.to_file(outfile)

    gdf = io_helpers._geopandas_read(outfile)
    assert isinstance(gdf, gpd.GeoDataFrame)


@pytest.mark.unittest
def test_geopandas_read_invalid_file(tmp_path):
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("This is not a valid geopandas file.")

    with pytest.raises(
        ValueError, match="File type .txt is not supported by geopandas."
    ):
        io_helpers._geopandas_read(invalid_file)


@pytest.mark.parametrize(
    "extension", [".pq", ".parquet", ".csv", ".txt", ".tsv", ".xls", ".xlsx"]
)
def test_pandas_read(tmp_path, point_header, extension):
    outfile = tmp_path / f"test{extension}"

    if extension in {".pq", ".parquet"}:
        point_header.to_parquet(outfile)
    elif extension in {".csv", ".txt", ".tsv"}:
        point_header.to_csv(outfile)
    elif extension in {".xls", ".xlsx"}:
        point_header.to_excel(outfile)

    df = io_helpers._pandas_read(outfile)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.unittest
def test_geost_data_to_collection(borehole_collection):
    collection = io_helpers._geost_data_to_collection(borehole_collection.data)
    assert hasattr(collection, "data")
    assert hasattr(collection, "header")


@pytest.mark.unittest
def test_geost_attrs_from_csv(testdatadir):
    csv_file = testdatadir / "test_borehole_table.csv"
    attrs = io_helpers._geost_attrs_from_csv(csv_file)
    assert isinstance(attrs, dict)
    assert "geost" in attrs
    assert "data_columns" in attrs["geost"]
    assert "header_columns" in attrs["geost"]
    assert attrs["geost"]["crs"] == "EPSG:28992"
    assert attrs["geost"]["vertical_datum"] == "EPSG:5709"
    assert "metadata" in attrs["geost"]


@pytest.mark.unittest
def test_to_geopackage(borehole_collection, tmp_path):
    outfile = tmp_path / "gpkg_with_error.gpkg"
    invalid_column = "GEOM"
    borehole_collection.header[invalid_column] = "test"
    with pytest.raises(FieldError):
        borehole_collection.to_geopackage(outfile)


@pytest.mark.unittest
def test_create_connection(simple_soilmap_gpkg):
    conn = io_helpers.create_connection(simple_soilmap_gpkg)
    assert isinstance(conn, sqlite3.Connection)
