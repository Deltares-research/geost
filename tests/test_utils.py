import sqlite3

import geopandas as gpd
import pandas as pd
import pytest
from pyogrio.errors import FieldError
from shapely import geometry as gmt

from geost import utils


@pytest.mark.unittest
def test_save_pickle(borehole_collection, tmp_path):
    outfile = tmp_path / r"collection.pkl"
    utils.save_pickle(borehole_collection, outfile)
    assert outfile.is_file()


@pytest.mark.unittest
def test_to_geopackage(borehole_collection, tmp_path):
    outfile = tmp_path / "gpkg_with_error.gpkg"
    invalid_column = "GEOM"
    borehole_collection.header[invalid_column] = "test"
    with pytest.raises(FieldError):
        borehole_collection.to_geopackage(outfile)


@pytest.mark.unittest
def test_create_connection(simple_soilmap_gpkg):
    conn = utils.create_connection(simple_soilmap_gpkg)
    assert isinstance(conn, sqlite3.Connection)


@pytest.mark.unittest
@pytest.mark.parametrize(
    "input_value,expected",
    [
        ("3.14", 3.14),
        ("0", 0.0),
        ("-2.5", -2.5),
        ("1e3", 1000.0),
        ("not_a_number", None),
        ("", None),
        (5, 5.0),
        (3.0, 3.0),
    ],
)
def test_safe_float(input_value, expected):
    result = utils.safe_float(input_value)
    assert (result == expected) or (result is None and expected is None)


@pytest.mark.unittest
def test_safe_float_invalid_type():
    """Test safe_float with an invalid type."""
    with pytest.raises(TypeError):
        utils.safe_float([])


@pytest.mark.unittest
def test_get_path_iterable_file(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("a,b\n1,2")
    result = utils.get_path_iterable(file)
    assert isinstance(result, list)
    assert result == [file]


@pytest.mark.unittest
def test_get_path_iterable_dir(tmp_path):
    (tmp_path / "a.txt").write_text("foo")
    (tmp_path / "b.txt").write_text("bar")
    result = list(utils.get_path_iterable(tmp_path, "*.txt"))
    assert len(result) == 2
    assert all(f.suffix == ".txt" for f in result)


@pytest.mark.unittest
def test_get_path_iterable_invalid(tmp_path):
    invalid_path = tmp_path / "does_not_exist"
    with pytest.raises(TypeError):
        utils.get_path_iterable(invalid_path)


@pytest.mark.parametrize(
    "geometry, expected_length",
    [
        (gpd.GeoDataFrame(geometry=[gmt.Point(1, 2)]), 1),
        (gmt.Point(1, 2), 1),
        (gmt.LineString([(0, 0), (1, 1)]), 1),
        (gmt.Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]), 1),
        (gmt.MultiPoint([gmt.Point(1, 2), gmt.Point(3, 4)]), 1),
        ([gmt.Point(1, 2), gmt.Point(3, 4)], 2),
    ],
)
def test_check_geometry_instance(geometry, expected_length):
    geom = utils.check_geometry_instance(geometry)
    assert isinstance(geom, gpd.GeoDataFrame)
    assert len(geom) == expected_length


@pytest.mark.unittest
def test_check_geometry_instance_from_file(tmp_path, point_header_gdf):
    outfile = tmp_path / "test.geoparquet"
    point_header_gdf.to_parquet(outfile)
    gdf = utils.check_geometry_instance(outfile)
    assert isinstance(gdf, gpd.GeoDataFrame)


@pytest.mark.parametrize("extension", [".geoparquet", ".parquet", ".gpkg", ".shp"])
def test_geopandas_read(tmp_path, point_header_gdf, extension):
    outfile = tmp_path / f"test{extension}"

    if extension in {".geoparquet", ".parquet"}:
        point_header_gdf.to_parquet(outfile)
    elif extension in {".gpkg", ".shp"}:
        point_header_gdf.to_file(outfile)

    gdf = utils._geopandas_read(outfile)
    assert isinstance(gdf, gpd.GeoDataFrame)


@pytest.mark.unittest
def test_geopandas_read_invalid_file(tmp_path):
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("This is not a valid geopandas file.")

    with pytest.raises(
        ValueError, match="File type .txt is not supported by geopandas."
    ):
        utils._geopandas_read(invalid_file)


@pytest.mark.parametrize(
    "extension", [".pq", ".parquet", ".csv", ".txt", ".tsv", ".xls", ".xlsx"]
)
def test_pandas_read(tmp_path, point_header_gdf, extension):
    outfile = tmp_path / f"test{extension}"

    if extension in {".pq", ".parquet"}:
        point_header_gdf.to_parquet(outfile)
    elif extension in {".csv", ".txt", ".tsv"}:
        point_header_gdf.to_csv(outfile)
    elif extension in {".xls", ".xlsx"}:
        point_header_gdf.to_excel(outfile)

    df = utils._pandas_read(outfile)
    assert isinstance(df, pd.DataFrame)
