import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from shapely import geometry as gmt

from geost._warnings import MixedDepthWarning
from geost.utils import conversion


@pytest.mark.unittest
def test_get_path_iterable_file(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("a,b\n1,2")
    result = conversion.get_path_iterable(file)
    assert isinstance(result, list)
    assert result == [file]


@pytest.mark.unittest
def test_get_path_iterable_dir(tmp_path):
    (tmp_path / "a.txt").write_text("foo")
    (tmp_path / "b.txt").write_text("bar")
    result = list(conversion.get_path_iterable(tmp_path, "*.txt"))
    assert len(result) == 2
    assert all(f.suffix == ".txt" for f in result)


@pytest.mark.unittest
def test_get_path_iterable_invalid(tmp_path):
    invalid_path = tmp_path / "does_not_exist"
    with pytest.raises(TypeError):
        conversion.get_path_iterable(invalid_path)


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
    result = conversion.safe_float(input_value)
    assert (result == expected) or (result is None and expected is None)


@pytest.mark.unittest
def test_safe_float_invalid_type():
    """Test safe_float with an invalid type."""
    with pytest.raises(TypeError):
        conversion.safe_float([])


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
    geom = conversion.check_geometry_instance(geometry)
    assert isinstance(geom, gpd.GeoDataFrame)
    assert len(geom) == expected_length


@pytest.mark.unittest
def test_check_geometry_instance_from_file(tmp_path, point_header):
    outfile = tmp_path / "test.geoparquet"
    point_header.to_parquet(outfile)
    gdf = conversion.check_geometry_instance(outfile)
    assert isinstance(gdf, gpd.GeoDataFrame)


@pytest.fixture
def layered_surface_positive_downward():
    return pd.DataFrame(
        {
            "nr": ["A", "A", "B", "B"],
            "surface": [0.2, 0.2, 0.3, 0.3],
            "top": [0, 0.5, 0, 0.4],
            "bottom": [0.5, 1.0, 0.4, 0.8],
        }
    )


@pytest.fixture
def layered_surface_negative_downward(borehole_data):
    negative_downward = borehole_data.copy()
    negative_downward[["top", "bottom"]] *= -1
    return negative_downward


@pytest.fixture
def layered_mixed_depth(borehole_data):
    mixed = borehole_data.copy()

    mask = mixed["nr"] == "B"
    mixed.loc[mask, "top"] = mixed["surface"] - mixed["top"]
    mixed.loc[mask, "bottom"] = mixed["surface"] - mixed["bottom"]
    return mixed


@pytest.fixture
def layered_nap(borehole_data):
    layered_nap = borehole_data.copy()
    layered_nap["top"] = layered_nap["surface"] - layered_nap["top"]
    layered_nap["bottom"] = layered_nap["surface"] - layered_nap["bottom"]
    return layered_nap


@pytest.fixture
def discrete_surface_negative_downward(cpt_data):
    negative_downward = cpt_data.copy()
    negative_downward["depth"] *= -1
    return negative_downward


@pytest.fixture
def discrete_nap(cpt_data):
    discrete_nap = cpt_data.copy()
    discrete_nap["depth"] = discrete_nap["surface"] - discrete_nap["depth"]
    return discrete_nap


@pytest.fixture
def discrete_mixed_depth(cpt_data):
    mixed = cpt_data.copy()

    mask = mixed["nr"] == "b"
    mixed.loc[mask, "depth"] = mixed["surface"] - mixed.loc[mask, "depth"]
    return mixed


@pytest.mark.parametrize(
    "df",
    [
        "borehole_data",
        "layered_surface_negative_downward",
        "layered_nap",
        "layered_mixed_depth",
    ],
    ids=[
        "layered_surface_positive_downward",
        "layered_surface_negative_downward",
        "layered_nap",
        "layered_mixed_depth",
    ],
)
def test_adjust_z_coordinates_layered(df, request):
    test_id = request.node.callspec.id

    df = request.getfixturevalue(df)

    if test_id == "layered_mixed_depth":
        with pytest.warns(MixedDepthWarning):
            result = conversion.adjust_z_coordinates(df.copy())
        assert_array_almost_equal(
            result[["top", "bottom"]], df[["top", "bottom"]]
        )  # Depth should not be adjusted for mixed depth data
    else:
        df = conversion.adjust_z_coordinates(df)
        assert_array_almost_equal(
            df[["top", "bottom"]],
            [
                [0.0, 0.8],
                [0.8, 1.5],
                [1.5, 2.5],
                [2.5, 3.7],
                [3.7, 4.2],
                [0.0, 0.6],
                [0.6, 1.2],
                [1.2, 2.5],
                [2.5, 3.1],
                [3.1, 3.9],
                [0.0, 1.4],
                [1.4, 1.8],
                [1.8, 2.9],
                [2.9, 3.8],
                [3.8, 5.5],
                [0.0, 0.5],
                [0.5, 1.2],
                [1.2, 1.8],
                [1.8, 2.5],
                [2.5, 3.0],
                [0.0, 0.5],
                [0.5, 1.2],
                [1.2, 1.8],
                [1.8, 2.5],
                [2.5, 3.0],
            ],
        )


@pytest.mark.parametrize(
    "df",
    [
        "cpt_data",
        "discrete_surface_negative_downward",
        "discrete_nap",
        "discrete_mixed_depth",
    ],
    ids=[
        "discrete_surface_positive_downward",
        "discrete_surface_negative_downward",
        "discrete_nap",
        "discrete_mixed_depth",
    ],
)
def test_adjust_z_coordinates_discrete(df, request):
    test_id = request.node.callspec.id

    df = request.getfixturevalue(df)
    df = conversion.adjust_z_coordinates(df)

    if test_id == "discrete_mixed_depth":
        with pytest.warns(MixedDepthWarning):
            result = conversion.adjust_z_coordinates(df.copy())
        assert_array_almost_equal(
            result["depth"], df["depth"]
        )  # Depth should not be adjusted for mixed depth data
    else:
        assert_array_almost_equal(
            df["depth"],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
            ],
        )


@pytest.mark.unittest
def test_adjust_z_coordinates_no_depth_columns(cpt_data):
    df = cpt_data.copy()
    df = df.drop(columns=["depth"])
    result = conversion.adjust_z_coordinates(df)
    assert result.equals(df)  # Should do nothing to the DataFrame and raise no error
