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
    "condition,left_operand_indexed_object,expected",
    [
        ("var1 < 10", None, "var1 < 10"),
        ("foo == 'bar'", None, "foo == 'bar'"),
        ("x!=42", None, "x != 42"),
        ("temp >= 0", "da", "da['temp'] >= 0"),
        ("y <= 5", "arr", "arr['y'] <= 5"),
    ],
)
def test_string_to_evaluable_valid(condition, left_operand_indexed_object, expected):
    result, var, val = utils.string_to_evaluable(condition, left_operand_indexed_object)
    assert result == expected


@pytest.mark.unittest
@pytest.mark.parametrize(
    "condition",
    [
        "invalid",
        "foo 10",
        "x <",
        "",
        "== 5",
    ],
)
def test_string_to_evaluable_invalid(condition):
    with pytest.raises(ValueError):
        utils.string_to_evaluable(condition)
