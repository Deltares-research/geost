import geopandas as gpd
import pytest
from shapely import geometry as gmt

from geost.utils import casting


@pytest.mark.unittest
def test_get_path_iterable_file(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("a,b\n1,2")
    result = casting.get_path_iterable(file)
    assert isinstance(result, list)
    assert result == [file]


@pytest.mark.unittest
def test_get_path_iterable_dir(tmp_path):
    (tmp_path / "a.txt").write_text("foo")
    (tmp_path / "b.txt").write_text("bar")
    result = list(casting.get_path_iterable(tmp_path, "*.txt"))
    assert len(result) == 2
    assert all(f.suffix == ".txt" for f in result)


@pytest.mark.unittest
def test_get_path_iterable_invalid(tmp_path):
    invalid_path = tmp_path / "does_not_exist"
    with pytest.raises(TypeError):
        casting.get_path_iterable(invalid_path)


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
    result = casting.safe_float(input_value)
    assert (result == expected) or (result is None and expected is None)


@pytest.mark.unittest
def test_safe_float_invalid_type():
    """Test safe_float with an invalid type."""
    with pytest.raises(TypeError):
        casting.safe_float([])


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
    geom = casting.check_geometry_instance(geometry)
    assert isinstance(geom, gpd.GeoDataFrame)
    assert len(geom) == expected_length


@pytest.mark.unittest
def test_check_geometry_instance_from_file(tmp_path, point_header):
    outfile = tmp_path / "test.geoparquet"
    point_header.to_parquet(outfile)
    gdf = casting.check_geometry_instance(outfile)
    assert isinstance(gdf, gpd.GeoDataFrame)
