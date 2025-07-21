import sqlite3

import pytest
from pyogrio.errors import FieldError

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
