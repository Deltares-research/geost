import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import pyvista as pv
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shapely.geometry import LineString, Polygon

from geost.accessor import GeostFrame
from geost.base import Collection
from geost.exceptions import (
    MissingDepthError,
    MissingGeometryError,
    MissingSurfaceError,
    MissingSurveyIDError,
    MissingXYError,
)
from geost.validation import column_names
from tests.conftest import cpt_data


@pytest.fixture
def dataframe():
    return pd.DataFrame({"nr": ["a", "b"], "x": [1, 2], "y": [3, 4]})


@pytest.fixture
def geodataframe(dataframe):
    return gpd.GeoDataFrame(
        dataframe, geometry=gpd.points_from_xy(dataframe.x, dataframe.y)
    )


@pytest.fixture
def test_polygon():
    return gpd.GeoDataFrame(
        {"id": [1], "letter": ["A"]},
        geometry=[Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))],
        crs=28992,
    )


class TestGeostFrame:
    @pytest.mark.unittest
    def test_accessor(self, dataframe: pd.DataFrame):
        assert hasattr(dataframe, "gst")
        assert isinstance(dataframe.gst, GeostFrame)

    @pytest.mark.unittest
    def test_set_positional_columns(self):
        names = {
            k: list(v) for k, v in column_names.POSITIONAL_COLUMN_NAMES.items()
        }  # Convert sets to lists because otherwise we can't index them

        longest_set = max(map(len, names.values()))

        for ii in range(longest_set):
            # Use every possible column name at least one time. Loop untill the longest
            # set of possible column names is exhausted. For each column type, we take
            # the ii-th name from the list of possible names, and if the list is shorter
            # than ii, we go back to the beginning of the list using modulo.
            nr = names["nr"][ii % len(names["nr"])]
            surface = names["surface"][ii % len(names["surface"])]
            end = names["end"][ii % len(names["end"])]
            x = names["x_coordinate"][ii % len(names["x_coordinate"])]
            y = names["y_coordinate"][ii % len(names["y_coordinate"])]
            top = names["top"][ii % len(names["top"])]
            bottom = names["depth"][ii % len(names["depth"])]

            df = pd.DataFrame(columns=[nr, surface, end, x, y, top, bottom])
            assert df.gst._nr == nr
            assert df.gst._surface == surface
            assert df.gst._end == end
            assert df.gst._x == x
            assert df.gst._y == y
            assert df.gst._top == top
            assert df.gst._bottom == bottom

        df = pd.DataFrame(
            columns=[
                "nr",
                "invalid_surface",
                "invalid_end",
                "invalid_x",
                "invalid_y",
                "invalid_top",
                "invalid_bottom",
            ]
        )
        assert df.gst._nr == "nr"
        assert df.gst._surface is None
        assert df.gst._end is None
        assert df.gst._x is None
        assert df.gst._y is None
        assert df.gst._top is None
        assert df.gst._bottom is None

        df = pd.DataFrame(columns=["nr"])
        assert df.gst._nr == "nr"

        df = pd.DataFrame(columns=["NR", "Latitude", "Longitude"])
        assert df.gst._nr == "NR"
        assert df.gst._x == "Longitude"
        assert df.gst._y == "Latitude"

        with pytest.raises(
            MissingSurveyIDError,
            match="DataFrame must contain a column identifying survey ID",
        ):
            df_invalid = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            df_invalid.gst

    @pytest.mark.parametrize(
        "df, top, bottom",
        [
            (pd.DataFrame(columns=["nr", "top", "bottom"]), "top", "bottom"),
            (pd.DataFrame(columns=["nr", "top", "bottom", "depth"]), "top", "bottom"),
            (pd.DataFrame(columns=["nr", "bottom"]), None, "bottom"),
            (pd.DataFrame(columns=["nr", "depth"]), None, "depth"),
            (pd.DataFrame(columns=["nr"]), None, None),  # No depth columns present
        ],
    )
    def test_has_depth_columns(self, df, top, bottom):
        if top is None:
            assert df.gst._top is None
        else:
            assert df.gst._top == top

        if bottom is None:
            assert df.gst._bottom is None
        else:
            assert df.gst._bottom == bottom

    @pytest.mark.unittest
    def test_has_geometry(
        self, dataframe: pd.DataFrame, geodataframe: gpd.GeoDataFrame
    ):
        assert not dataframe.gst.has_geometry
        assert geodataframe.gst.has_geometry

    @pytest.mark.parametrize(
        "df, x, y",
        [
            (pd.DataFrame(columns=["nr", "x", "y"]), "x", "y"),
            (pd.DataFrame(columns=["nr", "lon", "lat"]), "lon", "lat"),
            (
                pd.DataFrame(columns=["nr", "easting", "northing"]),
                "easting",
                "northing",
            ),
            (pd.DataFrame(columns=["nr", "x"]), "x", None),
            (pd.DataFrame(columns=["nr", "y"]), None, "y"),
            (pd.DataFrame(columns=["nr"]), None, None),
        ],
    )
    def test_has_xy_columns(self, df, x, y):
        if x is not None and y is not None:
            assert df.gst.has_xy_columns
            assert df.gst._x == x
            assert df.gst._y == y
        else:
            if x is not None:
                assert df.gst._x == x

            if y is not None:
                assert df.gst._y == y

            assert not df.gst.has_xy_columns

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame(columns=["nr", "surface", "top", "bottom"]),
            pd.DataFrame(columns=["nr", "surface", "bottom"]),
            pd.DataFrame(columns=["nr", "surface", "depth"]),
            pd.DataFrame(columns=["nr", "surface"]),
            pd.DataFrame(columns=["nr", "top", "bottom"]),
            pd.DataFrame(columns=["nr", "bottom"]),
            pd.DataFrame(columns=["nr", "depth"]),
        ],
        ids=[
            "surface-top-bottom",
            "surface-bottom",
            "surface-depth",
            "surface-missing-depth",
            "top-bottom-missing-surface",
            "bottom-missing-surface",
            "depth-missing-surface",
        ],
    )
    def test_has_depth_columns(self, df, request):
        test_id = request.node.callspec.id
        if test_id in {"surface-top-bottom", "surface-bottom", "surface-depth"}:
            assert df.gst.has_depth_columns
        else:
            assert not df.gst.has_depth_columns

    @pytest.mark.unittest
    def test_to_iterable(self, borehole_data):
        inst = "string"
        inst = borehole_data.gst._to_iterable(inst)
        assert isinstance(inst, list)

        inst = ["list of strings"]
        inst = borehole_data.gst._to_iterable(inst)
        assert isinstance(inst, list)

    @pytest.mark.unittest
    def test_first_row_survey(self, borehole_data):
        assert_array_equal(
            borehole_data.gst.first_row_survey,
            [
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        )

    @pytest.mark.unittest
    def test_last_row_survey(self, borehole_data):
        assert_array_equal(
            borehole_data.gst.last_row_survey,
            [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
            ],
        )

    @pytest.mark.unittest
    def test_to_header(self, borehole_data):
        header = borehole_data.gst.to_header()
        assert isinstance(header, gpd.GeoDataFrame)
        assert header.gst.has_geometry
        assert_array_equal(header.columns, ["nr", "x", "y", "surface", "geometry"])

        header = borehole_data.gst.to_header(
            include_columns=["top", "bottom", "lith"], coordinate_names=("x", "y")
        )
        assert isinstance(header, gpd.GeoDataFrame)
        assert header.gst.has_geometry
        assert_array_equal(
            header.columns,
            ["nr", "x", "y", "surface", "top", "bottom", "lith", "geometry"],
        )

        header = borehole_data.gst.to_header(coordinate_names=["x", "y"], crs=28992)
        assert isinstance(header, gpd.GeoDataFrame)
        assert header.gst.has_geometry
        assert header.crs == 28992

        with pytest.raises(
            KeyError,
            match="Coordinate columns 'missing_x' and/or 'missing_y' not found in DataFrame.",
        ):
            borehole_data.gst.to_header(coordinate_names=["missing_x", "missing_y"])

    @pytest.mark.unittest
    def test_to_collection(self, borehole_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            collection = borehole_data.gst.to_collection(
                coordinate_names=["x", "y"],
                crs=28992,
                vertical_datum=5709,
                has_inclined=True,
            )
        assert isinstance(collection, Collection)
        assert collection.has_inclined
        assert collection.header_has_geometry
        assert collection.crs == 28992
        assert collection.vertical_datum == 5709

        with pytest.warns() as record:
            borehole_data = borehole_data.drop(columns=["x", "y"])
            collection = borehole_data.gst.to_collection()

        # TODO: `record[0]` has unexpected validation warning is thrown, check
        assert ("Setting the header without an active geometry column.") in str(
            record[0].message
        )
        assert isinstance(collection, Collection)
        assert isinstance(collection.header, gpd.GeoDataFrame)
        assert isinstance(collection.data, pd.DataFrame)
        assert not collection.header_has_geometry
        assert collection.crs is None
        assert collection.vertical_datum is None

    @pytest.mark.unittest
    def test_standardize_column_names(self):
        df = pd.DataFrame(
            columns=[
                "nitg_nr",
                "maaiveld",
                "einddiepte",
                "easting",
                "northing",
                "tv_top_nap",
                "basis_diepte",
            ]
        )
        df.gst.standardize_column_names()
        assert_array_equal(
            df.columns,
            ["nr", "surface", "end", "x", "y", "top", "depth"],
        )

    @pytest.mark.unittest
    def test_determine_end_depth(self, borehole_data):
        # Test that the method correctly determines the end depth when "end" column is missing
        end = borehole_data.gst.determine_end_depth()
        assert_array_equal(end, borehole_data["end"])

    @pytest.mark.unittest
    def test_select_by_elevation(self, borehole_data):
        # Test with only top_min specified
        selected = borehole_data.gst.select_by_elevation(top_min=0, top_max=0.25)
        assert_array_equal(selected["nr"].unique(), ["A", "C", "D"])

        # Test with only end_min specified
        selected = borehole_data.gst.select_by_elevation(end_max=-3, end_min=-4)
        assert_array_equal(selected["nr"].unique(), ["A", "B", "E"])

        # Test same selection with end column missing, will be computed from data and added
        # to header, result must be the same.
        selected = borehole_data.drop(columns=["end"]).gst.select_by_elevation(
            end_max=-3, end_min=-4
        )
        assert_array_equal(selected["nr"].unique(), ["A", "B", "E"])

        expected_error = (
            "Cannot use 'end_min' and 'end_max' in select_by_elevation data"
            "has no column 'end' and no depth information is found in the data."
        )
        with pytest.raises(MissingDepthError, match=expected_error):
            borehole_data.drop(
                columns=["end", "top", "bottom"]
            ).gst.select_by_elevation(end_max=-3, end_min=-4)

        with pytest.raises(
            MissingSurfaceError,
            match="Method 'select_by_elevation' requires a surface column",
        ):
            borehole_data.rename(
                columns={"surface": "invalid_surface"}
            ).gst.select_by_elevation(top_min=0, top_max=0.25)

    @pytest.mark.unittest
    def test_select_by_length(self, borehole_data):
        selected = borehole_data.gst.select_by_length(max_length=3.0)
        assert_array_equal(selected["nr"].unique(), ["D", "E"])

        selected = borehole_data.gst.select_by_length(max_length=1.5)
        assert_array_equal(selected["nr"].unique(), [])

        selected = borehole_data.gst.select_by_length(min_length=4.0, max_length=5.0)
        assert_array_equal(selected["nr"].unique(), ["A"])

        selected = borehole_data.drop(columns=["end"]).gst.select_by_length(
            min_length=4.0, max_length=5.0
        )
        assert_array_equal(selected["nr"].unique(), ["A"])

        expected_error = (
            "Cannot use select_by_length if data has no column 'end' and no depth "
            "information is found in the data."
        )
        with pytest.raises(MissingDepthError, match=expected_error):
            borehole_data.drop(columns=["end", "top", "bottom"]).gst.select_by_length(
                min_length=4.0, max_length=5.0
            )

    @pytest.mark.unittest
    def test_to_crs(self, point_header):
        result = point_header.gst.to_crs(4326)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == 4326
        assert result["x"].between(3.31, 3.32).all()
        assert result["y"].between(47.97, 47.98).all()

        # Make sure "x" and "y" are not automatically computed when not present
        result = point_header.drop(columns=["x", "y"]).gst.to_crs(4326)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == 4326
        assert not {"x", "y"}.issubset(result.columns)

        with pytest.raises(
            TypeError,
            match="Method 'to_crs' requires a GeoDataFrame with a valid geometry column.",
        ):
            point_header.drop(columns="geometry").gst.to_crs(4326)

    @pytest.mark.skip("This test is for a method that is not yet implemented.")
    def test_to_vertical_datum(self, point_header):
        result = point_header.gst.to_vertical_datum(5710)
        assert isinstance(result, gpd.GeoDataFrame)
        assert np.isclose(result["surface"] - point_header["surface"], 2.28234).all()
        assert np.isclose(result["end"] - point_header["end"], 2.28234).all()

    @pytest.mark.unittest
    def test_transform_coordinates(self, nlog_borehole_collection):
        data = nlog_borehole_collection.data.iloc[:2]

        result = data.gst.transform_coordinates(28992, 4326, xbot="x_bot", ybot="y_bot")
        assert_array_almost_equal(result["x"], [3.51911014, 3.51925344])
        assert_array_almost_equal(result["x_bot"], [3.51925344, 3.5192852])
        assert_array_almost_equal(result["y"], [55.68101879, 55.68101195])
        assert_array_almost_equal(result["y_bot"], [55.68101195, 55.68101242])

        with pytest.raises(
            MissingXYError,
            match="Method 'transform_coordinates' requires x, y information in the DataFrame.",
        ):
            data.drop(columns=["x", "y"]).gst.transform_coordinates(28992, 4326)

    @pytest.mark.unittest
    def test_select_within_bbox(self, point_header):
        selected = point_header.gst.select_within_bbox(1, 1, 3, 3)
        assert len(selected) == 9

        # Test inverted selection: everything outside the bbox
        selected = point_header.gst.select_within_bbox(1, 1, 3, 3, invert=True)
        assert len(selected) == 16

        with pytest.raises(
            MissingGeometryError,
            match="Method 'select_within_bbox' requires a GeoDataFrame with a valid geometry column.",
        ):
            point_header = point_header.drop(columns="geometry")
            # Remove geometry column to make it invalid for spatial selection
            point_header.gst.select_within_bbox(1, 1, 3, 3)

    @pytest.mark.unittest
    def test_select_with_points(self, point_header):
        selection_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([1, 4, 1, 4], [1, 4, 4, 1]), crs=28992
        )
        max_distance = 1.1
        selected = point_header.gst.select_with_points(selection_points, max_distance)
        assert len(selected) == 16

        inverted_selection = point_header.gst.select_with_points(
            selection_points, max_distance, invert=True
        )
        assert len(inverted_selection) == 9

        # Select only the two nearest points
        selected = point_header.gst.select_with_points(
            selection_points, max_distance, n_points=2
        )
        assert len(selected) == 8

        inverted_selection = point_header.gst.select_with_points(
            selection_points, max_distance, invert=True, n_points=2
        )
        assert len(inverted_selection) == 17

        # Select with Shapely object
        selected = point_header.gst.select_with_points(
            selection_points["geometry"].iloc[0], max_distance
        )
        assert len(selected) == 3

        # Select with iterable
        selected = point_header.gst.select_with_points(
            selection_points["geometry"].to_list(), max_distance
        )
        assert len(selected) == 16

        with pytest.raises(MissingGeometryError):
            point_header = point_header.drop(columns="geometry")
            # Remove geometry column to make it invalid for spatial selection
            point_header.gst.select_with_points(selection_points, max_distance)

    @pytest.mark.unittest
    def test_select_with_lines(self, point_header):
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame(geometry=selection_lines, crs=28992)

        max_distance = 1
        selected = point_header.gst.select_with_lines(selection_gdf, max_distance)
        assert len(selected) == 21

        inverted_selection = point_header.gst.select_with_lines(
            selection_gdf, max_distance, invert=True
        )
        assert len(inverted_selection) == 4

        # Select with Shapely object
        selected = point_header.gst.select_with_lines(selection_lines[0], max_distance)
        assert len(selected) == 13

        # Select with iterable
        selected = point_header.gst.select_with_lines(selection_lines, max_distance)
        assert len(selected) == 21

        with pytest.raises(MissingGeometryError):
            point_header = point_header.drop(columns="geometry")
            # Remove geometry column to make it invalid for spatial selection
            point_header.gst.select_with_lines(selection_gdf, max_distance)

    @pytest.mark.unittest
    def test_select_within_polygons(self, point_header):
        selection_polygon = Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))
        selection_gdf = gpd.GeoDataFrame(geometry=[selection_polygon], crs=28992)

        # Geodataframe based selection
        selected = point_header.gst.select_within_polygons(selection_gdf)
        assert len(selected) == 3

        selected_inverted = point_header.gst.select_within_polygons(
            selection_gdf, invert=True
        )
        assert len(selected_inverted) == 22

        selected_buffered = point_header.gst.select_within_polygons(
            selection_gdf, buffer=0.1
        )
        assert len(selected_buffered) == 11

        selected_inverted_buffered = point_header.gst.select_within_polygons(
            selection_gdf, buffer=0.1, invert=True
        )
        assert len(selected_inverted_buffered) == 14

        # Shapely Polygon based selection
        selected = point_header.gst.select_within_polygons(selection_polygon)
        assert len(selected) == 3
        assert_array_equal(selected["nr"], ["nr7", "nr13", "nr19"])

        # Selection with Iterable
        selected = point_header.gst.select_within_polygons([selection_polygon])
        assert len(selected) == 3

        with pytest.raises(MissingGeometryError):
            point_header = point_header.drop(columns="geometry")
            # Remove geometry column to make it invalid for spatial selection
            point_header.gst.select_within_polygons(selection_gdf)

    @pytest.mark.unittest
    def test_find_point_pairs(self, point_header):
        selection_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([1, 4, 1, 4], [1, 4, 4, 1]), crs=28992
        )
        max_distance = 1.1

        # Normal test, so it lists the indices of points in the selection_points that are
        # within the max_distance of each point in the header.
        result = point_header.gst.find_point_pairs(
            selection_points, max_distance, return_distance=True
        )
        assert_array_equal(
            result,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [3.0, 2.0, 0.0],
                    [4.0, 2.0, 1.0],
                    [5.0, 0.0, 1.0],
                    [8.0, 2.0, 1.0],
                    [10.0, 3.0, 1.0],
                    [13.0, 1.0, 1.0],
                    [15.0, 3.0, 0.0],
                    [16.0, 3.0, 1.0],
                    [17.0, 1.0, 1.0],
                    [18.0, 1.0, 0.0],
                    [19.0, 1.0, 1.0],
                    [20.0, 3.0, 1.0],
                    [23.0, 1.0, 1.0],
                ]
            ),
        )

        # Reversed test, so it lists the indices of points in the header that are
        # within the max_distance of each point in the selection_points.
        selection_points["nr"] = ["A", "B", "C", "D"]
        result = selection_points.gst.find_point_pairs(
            point_header, max_distance + 1, return_distance=True, n_points=1
        )
        assert_array_almost_equal(
            result,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [2.0, 3.0, 0.0],
                    [2.0, 4.0, 1.0],
                    [0.0, 5.0, 1.0],
                    [0.0, 6.0, 1.41421356],
                    [2.0, 7.0, 1.41421356],
                    [2.0, 8.0, 1.0],
                    [2.0, 9.0, 1.41421356],
                    [3.0, 10.0, 1.0],
                    [3.0, 11.0, 1.41421356],
                    [1.0, 12.0, 1.41421356],
                    [1.0, 13.0, 1.0],
                    [1.0, 14.0, 1.41421356],
                    [3.0, 15.0, 0.0],
                    [3.0, 16.0, 1.0],
                    [1.0, 17.0, 1.0],
                    [1.0, 18.0, 0.0],
                    [1.0, 19.0, 1.0],
                    [3.0, 20.0, 1.0],
                    [3.0, 21.0, 1.41421356],
                    [1.0, 22.0, 1.41421356],
                    [1.0, 23.0, 1.0],
                    [1.0, 24.0, 1.41421356],
                ]
            ),
        )

    @pytest.mark.unittest
    def test_spatial_join(self, point_header, test_polygon, tmp_path):
        result = point_header.gst.spatial_join(test_polygon, label_id="id")
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.shape == (11, 7)
        assert_array_equal(
            result["nr"],
            [
                "nr2",
                "nr6",
                "nr7",
                "nr8",
                "nr12",
                "nr13",
                "nr14",
                "nr18",
                "nr19",
                "nr20",
                "nr24",
            ],
        )
        assert (result["id"] == 1).all()
        assert "letter" not in result.columns
        assert "index_right" not in result.columns  # We throw this away in the method

        result = point_header.gst.spatial_join(test_polygon, label_id=["id", "letter"])
        assert result.shape == (11, 8)
        assert (result["id"] == 1).all()
        assert (result["letter"] == "A").all()

        result = point_header.gst.spatial_join(test_polygon, label_id="id", how="left")
        assert len(result) == len(
            point_header
        )  # Left join should keep all original rows
        assert len(result.columns) == len(point_header.columns) + 1
        assert_array_equal(
            result["id"],
            [
                np.nan,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                np.nan,
            ],
        )

        # Test from a file path instead of a GeoDataFrame
        outfile = tmp_path / r"test_polygon.geoparquet"
        test_polygon.to_parquet(outfile)
        result = point_header.gst.spatial_join(outfile, label_id="id")
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.shape == (11, 7)
        assert "id" in result.columns

        # Test if the "label_id" column already exists in the original GeoDataFrame
        point_header["id"] = 0
        result = point_header.gst.spatial_join(test_polygon, label_id="id")
        assert result.shape == (11, 8)
        assert (result["id_left"] == 0).all()
        assert (result["id_right"] == 1).all()

        result = point_header.gst.spatial_join(
            test_polygon, label_id="id", drop_label_if_exists=True
        )
        assert result.shape == (11, 7)
        assert (result["id"] == 1).all()

        with pytest.raises(
            MissingGeometryError,
            match="Method 'spatial_join' requires a GeoDataFrame with a valid geometry column.",
        ):
            point_header = point_header.drop(columns="geometry")
            point_header.gst.spatial_join(test_polygon, label_id="id")

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_data):
        selected = borehole_data.gst.select_by_values("lith", ["V", "K"], how="or")
        assert isinstance(selected, pd.DataFrame)
        assert_array_equal(selected["nr"].unique(), ["A", "B", "C", "D"])
        assert selected.shape == (20, 8)

        selected = borehole_data.gst.select_by_values("lith", ["V", "K"], how="and")
        assert_array_equal(selected["nr"].unique(), ["B", "D"])
        assert selected.shape == (10, 8)

        # Test inverted selection of the previous test case
        selected = borehole_data.gst.select_by_values(
            "lith", ["V", "K"], how="and", invert=True
        )
        assert_array_equal(selected["nr"].unique(), ["A", "C", "E"])
        assert selected.shape == (15, 8)

        selected = borehole_data.gst.select_by_values("bottom", slice(3.9, 10))
        assert_array_equal(selected["nr"].unique(), ["A", "B", "C"])
        assert selected.shape == (15, 8)

        selected = borehole_data.gst.select_by_values(
            "bottom", slice(3.9, 10), inclusive="neither"
        )
        assert_array_equal(selected["nr"].unique(), ["A", "C"])
        assert selected.shape == (10, 8)

        selected = borehole_data.gst.select_by_values("bottom", slice(None, None))
        assert selected.shape == borehole_data.shape

        with pytest.raises(TypeError, match="Unsupported type of selection values"):
            borehole_data.gst.select_by_values("lith", {"a": "V"})

        with pytest.raises(
            TypeError, match="Can only use a slice selection on numerical columns."
        ):
            borehole_data.gst.select_by_values("lith", slice(0, 1))

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_data):
        sliced = borehole_data.gst.slice_by_values("lith", "Z")
        assert isinstance(sliced, pd.DataFrame)
        assert_array_equal(sliced.index, [2, 3, 13, 14, 19, 20, 21, 22, 23, 24])
        assert (sliced["lith"] == "Z").all()

        # Test slicing everything except "Z"
        sliced = borehole_data.gst.slice_by_values("lith", "Z", invert=True)
        assert_array_equal(
            sliced.index, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18]
        )
        assert not (sliced["lith"] == "Z").any()

        # Test using a list of values
        sliced = borehole_data.gst.slice_by_values("lith", ["V", "K"])
        assert_array_equal(
            sliced.index, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18]
        )
        assert not ~(sliced["lith"].isin(["V", "K"])).any()

        sliced = borehole_data.gst.slice_by_values("lith", ["V", "K"], invert=True)
        assert_array_equal(sliced.index, [2, 3, 13, 14, 19, 20, 21, 22, 23, 24])
        assert (sliced["lith"] == "Z").all()

        sliced = borehole_data.gst.slice_by_values("bottom", slice(1.5, 3.1))
        assert_array_equal(sliced.index, [1, 2, 7, 8, 11, 12, 17, 18, 19, 22, 23, 24])
        assert sliced["bottom"].between(1.5, 3.1).all()

        sliced = borehole_data.gst.slice_by_values(
            "bottom", slice(1.5, 3.1), invert=True
        )
        assert_array_equal(sliced.index, [0, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 20, 21])
        assert not sliced["bottom"].between(1.5, 3.1).any()

        with pytest.raises(TypeError, match="Unsupported type of selection values"):
            borehole_data.gst.slice_by_values("lith", {"a": "V"})

        sliced = borehole_data.gst.slice_by_values(
            "bottom", slice(None, 3.1), inclusive="neither"
        )
        assert_array_equal(
            sliced.index,
            [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        )
        assert (sliced["bottom"] < 3.1).all()

        sliced = borehole_data.gst.slice_by_values(
            "bottom", slice(3.1, None), inclusive="neither"
        )
        assert_array_equal(sliced.index, [3, 4, 9, 13, 14])
        assert (sliced["bottom"] > 3.1).all()

        with pytest.raises(
            TypeError, match="Can only use a slice selection on numerical columns."
        ):
            borehole_data.gst.slice_by_values("lith", slice(0, 1))

    @pytest.mark.unittest
    def test_select_by_condition(self, borehole_data):
        selected = borehole_data.gst.select_by_condition(borehole_data["lith"] == "V")
        assert isinstance(selected, pd.DataFrame)

        expected_nrs = ["B", "D"]
        assert_array_equal(selected["nr"].unique(), expected_nrs)
        assert (selected["lith"] == "V").all()
        assert len(selected) == 4

        selected = borehole_data.gst.select_by_condition(
            borehole_data["lith"] == "V", invert=True
        )
        assert len(selected) == 21
        assert not (selected["lith"] == "V").all()

    @pytest.mark.unittest
    def test_slice_depth_interval_layered(self, borehole_data):
        # Test slicing with respect to depth below the surface.
        upper, lower = 0.6, 2.4
        sliced = borehole_data.gst.slice_depth_interval(upper, lower)
        assert isinstance(sliced, pd.DataFrame)

        layers_per_borehole = sliced["nr"].value_counts()
        expected_layer_count = [3, 3, 3, 3, 2]

        assert len(sliced) == 14
        assert sliced["top"].min() == upper
        assert sliced["bottom"].max() == lower
        assert_array_equal(layers_per_borehole, expected_layer_count)

        # Test slicing without updating layer boundaries.
        sliced = borehole_data.gst.slice_depth_interval(
            upper, lower, update_layer_boundaries=False
        )

        expected_tops_of_slice = [0.0, 0.6, 0.0, 0.5, 0.5]
        expected_bottoms_of_slice = [2.5, 2.5, 2.9, 2.5, 2.5]

        tops_of_slice = sliced.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.groupby("nr")["bottom"].max()

        assert len(sliced) == 14
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        # Test slicing with respect to a vertical reference plane.
        nap_upper, nap_lower = -2, -3
        sliced = borehole_data.gst.slice_depth_interval(
            nap_upper, nap_lower, relative_to_vertical_reference=True
        )

        expected_tops_of_slice = [2.2, 2.3, 2.25, 2.1, 1.9]
        expected_bottoms_of_slice = [3.2, 3.3, 3.25, 3.0, 2.9]

        tops_of_slice = sliced.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.groupby("nr")["bottom"].max()

        assert len(sliced) == 11
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        # Test slices that return empty objects.
        empty_slice = borehole_data.gst.slice_depth_interval(-2, -1)
        empty_slice_nap = borehole_data.gst.slice_depth_interval(
            3, 2, relative_to_vertical_reference=True
        )

        assert len(empty_slice) == 0
        assert len(empty_slice_nap) == 0

        # Test slicing using only an upper boundary or lower boundary.
        upper = 4
        sliced = borehole_data.gst.slice_depth_interval(upper)

        expected_boreholes = ["A", "C"]

        assert len(sliced) == 2
        assert_array_equal(sliced["nr"], expected_boreholes)

        nap_lower = -0.5
        sliced = borehole_data.gst.slice_depth_interval(
            lower_boundary=nap_lower, relative_to_vertical_reference=True
        )

        bottoms_of_slice = sliced.groupby("nr")["bottom"].max()
        expected_bottoms_of_slice = [0.7, 0.8, 0.75, 0.6, 0.4]

        assert len(sliced) == 7
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        with pytest.raises(
            MissingDepthError,
            match="Method 'slice_depth_interval' requires depth information in the DataFrame.",
        ):
            borehole_data_no_depth = borehole_data.drop(columns=["top", "bottom"])
            borehole_data_no_depth.gst.slice_depth_interval(0, 1)

    @pytest.mark.unittest
    def test_slice_depth_interval_discrete(self, cpt_data):
        upper, lower = 0.6, 4.4
        sliced = cpt_data.gst.slice_depth_interval(upper, lower)

        assert_array_equal(
            sliced["depth"], [1.0, 2.0, 3.0, 4.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        )
        assert_array_equal(sliced.index, [0, 1, 2, 3, 11, 12, 13, 14, 15, 16, 17])

        upper, lower = 0.1, -3.2
        sliced = cpt_data.gst.slice_depth_interval(
            upper, lower, relative_to_vertical_reference=True
        )
        assert_array_equal(
            sliced["depth"], [2.0, 3.0, 4.0, 5.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        )
        assert_array_equal(sliced.index, [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17])

        # Selection with respect to surface level using one limit
        sliced = cpt_data.gst.slice_depth_interval(lower_boundary=1.9)
        assert_array_equal(sliced["depth"], [1.0, 0.5, 1.0, 1.5])
        assert_array_equal(sliced.index, [0, 10, 11, 12])

        # Selection with respect to vertical reference plane using one limit
        sliced = cpt_data.gst.slice_depth_interval(
            lower_boundary=-0.1, relative_to_vertical_reference=True
        )
        assert_array_equal(sliced["depth"], [1.0, 2.0, 0.5])
        assert_array_equal(sliced.index, [0, 1, 10])

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "grid, reference, update, expected_bottoms",
        [
            ("nap_grid", True, True, [0.8, 1.0, 0.6, 0.8, 1.05, 0.5, 0.6]),
            ("depth_grid", False, True, [0.8, 0.5, 0.8, 0.5, 0.7]),
            ("nap_grid", True, False, [0.8, 1.5, 0.6, 1.2, 1.4, 0.5, 1.2]),
            ("depth_grid", False, False, [0.8, 0.6, 1.4, 0.5, 1.2]),
        ],
    )
    def test_slice_depth_interval_with_grid(
        self,
        borehole_data,
        grid: Literal["nap_grid"] | Literal["depth_grid"],
        reference: bool,
        update: bool,
        expected_bottoms: list | None,
        request: pytest.FixtureRequest,
    ):
        lower_boundary = request.getfixturevalue(grid)
        sliced = borehole_data.gst.slice_depth_interval(
            lower_boundary=lower_boundary,
            relative_to_vertical_reference=reference,
            update_layer_boundaries=update,
        )
        assert_array_almost_equal(
            sliced["bottom"].tolist(),
            expected_bottoms,
        )

    @pytest.mark.unittest
    def test_calculate_thickness_layered(self, borehole_data):
        result = borehole_data.gst.calculate_thickness()
        assert isinstance(result, pd.Series)
        assert_array_almost_equal(
            result,
            [
                0.8,
                0.7,
                1.0,
                1.2,
                0.5,
                0.6,
                0.6,
                1.3,
                0.6,
                0.8,
                1.4,
                0.4,
                1.1,
                0.9,
                1.7,
                0.5,
                0.7,
                0.6,
                0.7,
                0.5,
                0.5,
                0.7,
                0.6,
                0.7,
                0.5,
            ],
        )
        with pytest.raises(
            MissingDepthError,
            match="'calculate_thickness' requires at least bottom information of layers",
        ):
            borehole_data.drop(columns=["top", "bottom"]).gst.calculate_thickness()

    @pytest.mark.unittest
    def test_calculate_thickness_discrete(self, cpt_data):
        result = cpt_data.gst.calculate_thickness()
        assert isinstance(result, pd.Series)
        assert_array_almost_equal(
            result,
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        )

        with pytest.raises(
            MissingDepthError,
            match="'calculate_thickness' requires at least bottom information of layers",
        ):
            cpt_data.drop(columns=["depth"]).gst.calculate_thickness()

    @pytest.mark.unittest
    def test_cumulative_thickness_layered(self, borehole_data):
        result = borehole_data.gst.get_cumulative_thickness("lith", "V")
        assert (
            "thickness" not in borehole_data.columns
        )  # Ensure thickness column is not added to original DataFrame
        assert isinstance(result, pd.Series)
        assert_array_equal(result.index, ["B", "D"])
        assert_array_almost_equal(result, [1.9, 1.4])

        result = borehole_data.gst.get_cumulative_thickness("lith", ["Z", "K"])
        assert isinstance(result, pd.Series)
        assert_array_equal(result.index, ["A", "B", "C", "D", "E"])
        assert_array_almost_equal(result, [4.2, 2.0, 5.5, 1.6, 3.0])

    @pytest.mark.unittest
    def test_cumulative_thickness_discrete(self, cpt_data):
        result = cpt_data.gst.get_cumulative_thickness("qc", slice(0.7, 13))
        assert (
            "thickness" not in cpt_data.columns
        )  # Ensure thickness column is not added to original DataFrame
        assert_array_equal(result.index, ["a", "b"])
        assert_array_equal(result, [2, 1])

    @pytest.mark.unittest
    def test_get_layer_top_layered(self, borehole_data):
        result = borehole_data.gst.get_layer_top("lith", "V")
        assert isinstance(result, pd.Series)
        assert (
            "thickness" not in borehole_data.columns
        )  # Ensure thickness column is not added to original DataFrame
        assert_array_equal(result.index, ["B", "D"])
        assert_array_almost_equal(result, [1.2, 0.5])

        result = borehole_data.gst.get_layer_top("lith", "V", min_thickness=1.0)
        assert_array_equal(result.index, ["B"])
        assert_array_almost_equal(result, [1.2])

        result = borehole_data.gst.get_layer_top("lith", ["Z", "V"])
        assert_array_equal(result.index, ["A", "B", "C", "D", "E"])
        assert_array_almost_equal(result, [1.5, 1.2, 2.9, 0.5, 0.0])

        # Internally result and other are calculated in different ways, but should give the same result when using the same parameters
        result = borehole_data.gst.get_layer_top("lith", ["Z", "V"], min_thickness=1.0)
        other = borehole_data.gst.get_layer_top(
            "lith", ["Z", "V"], min_thickness=1.0, min_fraction=1.0
        )
        assert result.equals(other)

        result = borehole_data.gst.get_layer_top("bottom", slice(1.5, 3.1))
        assert_array_equal(result.index, ["A", "B", "C", "D", "E"])
        assert_array_almost_equal(result, [0.8, 1.2, 1.4, 1.2, 1.2])

    @pytest.mark.unittest
    def test_get_layer_top_discrete(self, cpt_data):
        result = cpt_data.gst.get_layer_top("qc", slice(0.7, 18))
        assert isinstance(result, pd.Series)
        assert (
            "thickness" not in cpt_data.columns
        )  # Ensure thickness column is not added to original DataFrame
        assert_array_equal(result.index, ["a", "b"])
        assert_array_equal(result, [8.0, 0.0])

        result = cpt_data.gst.get_layer_top("qc", slice(0.7, 18), min_thickness=2.5)
        assert_array_equal(result.index, ["b"])
        assert_array_equal(result, [0.0])

    @pytest.mark.unittest
    def test_get_discretization_value_layered(self, borehole_data):
        """
        Test for the `_get_discretization` helper method for `compute_discretized_fractions`
        when the input is a value.

        """
        discretization = borehole_data.gst._get_discretization(
            2.0, relative_to_reference=False
        )
        assert isinstance(discretization, pd.DataFrame)
        assert_array_equal(
            discretization["nr"], np.repeat(borehole_data["nr"].unique(), 3)
        )
        assert_array_equal(
            discretization["surface"], np.repeat(borehole_data["surface"].unique(), 3)
        )
        assert_array_equal(
            discretization["top"], np.tile([0, 2, 4], len(borehole_data["nr"].unique()))
        )
        assert_array_equal(
            discretization["bottom"],
            np.tile([2, 4, 6], len(borehole_data["nr"].unique())),
        )
        assert_array_equal(
            discretization["nz"], np.tile([0, 1, 2], len(borehole_data["nr"].unique()))
        )
        assert (discretization["dz"] == 2.0).all()

        discretization = borehole_data.gst._get_discretization(
            2.0, relative_to_reference=True
        )
        assert isinstance(discretization, pd.DataFrame)
        assert_array_equal(
            discretization["nr"], np.repeat(borehole_data["nr"].unique(), 4)
        )
        assert_array_equal(
            discretization["surface"], np.repeat(borehole_data["surface"].unique(), 4)
        )
        assert_array_equal(
            discretization["top"],
            np.tile([2, 0, -2, -4], len(borehole_data["nr"].unique())),
        )
        assert_array_equal(
            discretization["bottom"],
            np.tile([0, -2, -4, -6], len(borehole_data["nr"].unique())),
        )
        assert_array_equal(
            discretization["nz"],
            np.tile([0, 1, 2, 3], len(borehole_data["nr"].unique())),
        )
        assert (discretization["dz"] == 2.0).all()

    @pytest.mark.unittest
    def test_get_discretization_value_discrete(self, cpt_data):
        discretization = cpt_data.gst._get_discretization(
            2.0, relative_to_reference=False
        )
        assert isinstance(discretization, pd.DataFrame)
        assert "top" not in discretization.columns
        assert_array_equal(discretization["nr"], np.repeat(cpt_data["nr"].unique(), 5))
        assert_array_equal(
            discretization["surface"], np.repeat(cpt_data["surface"].unique(), 5)
        )
        assert_array_equal(
            discretization["depth"],
            np.tile([2, 4, 6, 8, 10], len(cpt_data["nr"].unique())),
        )
        assert_array_equal(
            discretization["nz"], np.tile([0, 1, 2, 3, 4], len(cpt_data["nr"].unique()))
        )
        assert (discretization["dz"] == 2.0).all()

        discretization = cpt_data.gst._get_discretization(
            2.0, relative_to_reference=True
        )
        assert "top" not in discretization.columns
        assert isinstance(discretization, pd.DataFrame)
        assert_array_equal(discretization["nr"], np.repeat(cpt_data["nr"].unique(), 6))
        assert_array_equal(
            discretization["surface"], np.repeat(cpt_data["surface"].unique(), 6)
        )
        assert_array_equal(
            discretization["depth"],
            np.tile([2, 0, -2, -4, -6, -8], len(cpt_data["nr"].unique())),
        )
        assert_array_equal(
            discretization["nz"],
            np.tile([0, 1, 2, 3, 4, 5], len(cpt_data["nr"].unique())),
        )
        assert (discretization["dz"] == 2.0).all()

    @pytest.mark.unittest
    def test_get_discretization_array_layered(self, borehole_data):
        """
        Test for the `_get_discretization` helper method for `compute_discretized_fractions`
        when the input is an array or list.

        """
        absolute_depth = [0.5, 1.0, 2.0]
        nap_depth = [1, 0, -1]

        # Using layered data, absolute depth
        discretization = borehole_data.gst._get_discretization(
            absolute_depth, relative_to_reference=False
        )
        assert isinstance(discretization, pd.DataFrame)
        assert_array_equal(
            discretization["nr"],
            np.repeat(borehole_data["nr"].unique(), len(absolute_depth)),
        )
        assert_array_equal(
            discretization["surface"],
            np.repeat(borehole_data["surface"].unique(), len(absolute_depth)),
        )
        assert_array_almost_equal(
            discretization["top"],
            np.tile([0.0, 0.5, 1.0], len(borehole_data["nr"].unique())),
        )
        assert_array_almost_equal(
            discretization["bottom"],
            np.tile(absolute_depth, len(borehole_data["nr"].unique())),
        )
        assert_array_almost_equal(
            discretization["dz"],
            np.tile([0.5, 0.5, 1.0], len(borehole_data["nr"].unique())),
        )
        assert_array_almost_equal(
            discretization["nz"],
            np.tile([0, 1, 2], len(borehole_data["nr"].unique())),
        )

        # Using layered data, NAP depth
        discretization = borehole_data.gst._get_discretization(
            nap_depth, relative_to_reference=True
        )
        assert_array_equal(
            discretization["nr"],
            np.repeat(borehole_data["nr"].unique(), len(nap_depth) - 1),
        )
        assert_array_equal(
            discretization["surface"],
            np.repeat(borehole_data["surface"].unique(), len(nap_depth) - 1),
        )
        assert_array_almost_equal(
            discretization["top"],
            np.tile([1, 0], len(borehole_data["nr"].unique())),
        )
        assert_array_almost_equal(
            discretization["bottom"],
            np.tile([0, -1], len(borehole_data["nr"].unique())),
        )
        assert (discretization["dz"] == 1).all()
        assert_array_almost_equal(
            discretization["nz"],
            np.tile([0, 1], len(borehole_data["nr"].unique())),
        )

    @pytest.mark.unittest
    def test_get_discretization_array_discrete(self, cpt_data):
        absolute_depth = [0.5, 1.0, 2.0]
        nap_depth = [1, 0, -1]

        # Using discrete data, absolute depth
        discretization = cpt_data.gst._get_discretization(
            absolute_depth, relative_to_reference=False
        )
        assert "top" not in discretization.columns
        assert_array_equal(
            discretization["nr"],
            np.repeat(cpt_data["nr"].unique(), len(absolute_depth)),
        )
        assert_array_equal(
            discretization["surface"],
            np.repeat(cpt_data["surface"].unique(), len(absolute_depth)),
        )
        assert_array_almost_equal(
            discretization["depth"],  # Name of bottom follows that of the name in CPTs
            np.tile(absolute_depth, len(cpt_data["nr"].unique())),
        )
        assert_array_almost_equal(
            discretization["dz"],
            np.tile([0.5, 0.5, 1.0], len(cpt_data["nr"].unique())),
        )
        assert_array_almost_equal(
            discretization["nz"],
            np.tile([0, 1, 2], len(cpt_data["nr"].unique())),
        )

        # Using discrete data, NAP depth
        discretization = cpt_data.gst._get_discretization(
            nap_depth, relative_to_reference=True
        )
        assert "top" not in discretization.columns
        assert_array_equal(
            discretization["nr"],
            np.repeat(cpt_data["nr"].unique(), len(nap_depth) - 1),
        )
        assert_array_equal(
            discretization["surface"],
            np.repeat(cpt_data["surface"].unique(), len(nap_depth) - 1),
        )
        assert_array_almost_equal(
            discretization["depth"],  # Name of bottom follows that of the name in CPTs
            np.tile(nap_depth[1:], len(cpt_data["nr"].unique())),
        )
        assert (discretization["dz"] == 1).all()
        assert_array_almost_equal(
            discretization["nz"],
            np.tile([0, 1], len(cpt_data["nr"].unique())),
        )

    @pytest.fixture
    def discretization_df(self):
        return pd.DataFrame(
            {
                "nr": np.repeat(["A", "B"], [3, 4]),
                "surface": np.repeat([0.4, 0.2], [3, 4]),
                "top": [0, 0.5, 1.0, 0, 0.5, 1.0, 2.0],
                "bottom": [0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 4.0],
            }
        )

    @pytest.mark.unittest
    def test_get_discretization_from_dataframe(self, borehole_data, discretization_df):
        """
        Test for the `_get_discretization` helper method for `compute_discretized_fractions`
        when the input is a DataFrame.

        """
        expected_thickness = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 2.0]
        expected_nz = [0, 1, 2, 3, 4, 5, 6]

        # Discretization contains "surface", "top" and "bottom" with absolute depth. This
        # corrects the absolute depths based on the surface difference, adds the applied
        # correction as a "surface_correction" column.
        dc = borehole_data.gst._get_discretization(
            discretization_df, relative_to_reference=False
        )
        assert isinstance(dc, pd.DataFrame)
        assert_array_equal(dc["nr"], np.repeat(["A", "B"], [3, 4]))
        assert_array_almost_equal(dc["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(dc["top"], [0.0, 0.3, 0.8, 0.0, 0.6, 1.1, 2.1])
        assert_array_almost_equal(dc["bottom"], [0.3, 0.8, 1.8, 0.6, 1.1, 2.1, 4.1])
        assert_array_almost_equal(
            dc["surface_correction"], np.repeat([-0.2, 0.1], [3, 4])
        )
        assert_array_almost_equal(dc["dz"], [0.3, 0.5, 1.0, 0.6, 0.5, 1.0, 2.0])
        assert_array_equal(dc["nz"], expected_nz)

        # Discretization only contains "bottom" with absolute depth
        dc = borehole_data.gst._get_discretization(
            discretization_df.drop(columns=["surface", "top"]),
            relative_to_reference=False,
        )
        assert isinstance(dc, pd.DataFrame)
        assert_array_equal(dc["nr"], np.repeat(["A", "B"], [3, 4]))
        assert_array_equal(dc["bottom"], discretization_df["bottom"])
        assert_array_almost_equal(dc["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(dc["dz"], expected_thickness)
        assert_array_equal(dc["nz"], expected_nz)

        # Discretization only contains "top" and "bottom" with absolute depth
        dc = borehole_data.gst._get_discretization(
            discretization_df.drop(columns="surface"),
            relative_to_reference=False,
        )
        assert isinstance(dc, pd.DataFrame)
        assert_array_almost_equal(dc["dz"], expected_thickness)
        assert_array_equal(dc["nr"], np.repeat(["A", "B"], [3, 4]))
        assert_array_equal(dc["top"], discretization_df["top"])
        assert_array_equal(dc["bottom"], discretization_df["bottom"])
        assert_array_almost_equal(dc["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(dc["dz"], expected_thickness)
        assert_array_equal(dc["nz"], expected_nz)

        with pytest.raises(
            MissingDepthError,
            match="The discretization DataFrame is missing bottom depths of layers.",
        ):
            borehole_data.gst._get_discretization(
                discretization_df.drop(columns="bottom"), False
            )

    @pytest.mark.unittest
    def test_get_discretization_from_dataframe_nap(
        self, borehole_data, discretization_df
    ):
        nap_df = discretization_df.gst._get_depth_relative_to_surface()

        expected_thickness = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 2.0]
        expected_nz = [0, 1, 2, 3, 4, 5, 6]

        # Discretization contains "surface", "top" and "bottom" with depth to NAP
        dc = borehole_data.gst._get_discretization(nap_df, relative_to_reference=True)
        assert_array_equal(dc["nr"], np.repeat(["A", "B"], [3, 4]))
        assert_array_almost_equal(dc["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(dc["top"], nap_df["top"])
        assert_array_almost_equal(dc["bottom"], nap_df["bottom"])
        assert_array_almost_equal(dc["dz"], expected_thickness)
        assert_array_equal(dc["nz"], expected_nz)

        # Discretization contains "top" and "bottom" with depth to NAP
        dc = borehole_data.gst._get_discretization(
            nap_df.drop(columns="surface"), relative_to_reference=True
        )
        assert_array_equal(dc["nr"], np.repeat(["A", "B"], [3, 4]))
        assert_array_almost_equal(dc["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(dc["top"], nap_df["top"])
        assert_array_almost_equal(dc["bottom"], nap_df["bottom"])
        assert_array_almost_equal(dc["dz"], expected_thickness)
        assert_array_equal(dc["nz"], expected_nz)

        # Discretization contains "surface", "bottom" with depth to NAP, this results in
        # a different layer thickness of the first layers of both surveys because of the
        # surface level difference between borehole_data and discretization_df
        dc = borehole_data.gst._get_discretization(
            nap_df.drop(columns="top"), relative_to_reference=True
        )
        assert_array_equal(dc["nr"], np.repeat(["A", "B"], [3, 4]))
        assert_array_almost_equal(dc["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(dc["bottom"], nap_df["bottom"])
        assert_array_almost_equal(dc["dz"], [0.3, 0.5, 1.0, 0.6, 0.5, 1.0, 2.0])
        assert_array_equal(dc["nz"], expected_nz)

        with pytest.raises(
            MissingDepthError,
            match="The discretization DataFrame is missing bottom depths of layers.",
        ):
            borehole_data.gst._get_discretization(
                nap_df.drop(columns="bottom"), relative_to_reference=True
            )

        with pytest.raises(
            MissingDepthError,
            match="The discretization DataFrame is missing surface information and top depth",
        ):
            borehole_data.gst._get_discretization(
                nap_df.drop(columns=["surface", "top"]), relative_to_reference=True
            )

    @pytest.mark.unittest
    def test_compute_discretized_fractions_with_bins(self, borehole_data):
        rand_rng = np.random.default_rng(seed=12)

        discretization = np.array([0.5, 1.0, 2.0, 4.0])
        subset = borehole_data.gst.select_by_values("nr", ["A", "B"])
        subset["value"] = rand_rng.random(len(subset))

        result = subset.gst.compute_discretized_fractions(
            "value", discretization, breaks=[0.33, 0.67]
        )
        assert isinstance(result, pd.DataFrame)
        assert_array_equal(result["nr"], ["A", "A", "A", "A", "B", "B", "B", "B"])
        assert_array_equal(result["surface"], [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3])
        assert_array_almost_equal(
            result["top"], [0.0, 0.5, 1.0, 2.0, 0.0, 0.5, 1.0, 2.0]
        )
        assert_array_almost_equal(
            result["bottom"], [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0]
        )
        assert_array_almost_equal(
            result["dz"], [0.5, 0.5, 1.0, 2.0, 0.5, 0.5, 1.0, 2.0]
        )
        assert_array_almost_equal(
            result["<=0.33"], [1.0, 0.6, 0.5, 0.85, 1.0, 0.2, 0.8, 0.25]
        )
        assert_array_almost_equal(
            result["0.33-0.67"],
            [np.nan, np.nan, np.nan, 0.15, np.nan, np.nan, np.nan, np.nan],
        )
        assert_array_almost_equal(
            result[">0.67"], [np.nan, 0.4, 0.5, np.nan, np.nan, 0.8, 0.2, 0.7]
        )

        # If we specify breaks that cover the entire range, only the column names should be changed
        result_entire_range = subset.gst.compute_discretized_fractions(
            "value", discretization, breaks=[0, 0.33, 0.67, 1]
        )
        assert_array_equal(
            result_entire_range.columns,
            ["nr", "surface", "top", "bottom", "dz", "0-0.33", "0.33-0.67", "0.67-1"],
        )
        assert result[["nr", "surface", "top", "bottom", "dz"]].equals(
            result_entire_range[["nr", "surface", "top", "bottom", "dz"]]
        )
        assert_array_almost_equal(result["<=0.33"], result_entire_range["0-0.33"])
        assert_array_almost_equal(result["0.33-0.67"], result_entire_range["0.33-0.67"])
        assert_array_almost_equal(result[">0.67"], result_entire_range["0.67-1"])

        # Test using only one value as "break"
        result_single_break = subset.gst.compute_discretized_fractions(
            "value", discretization, breaks=0.5
        )
        assert result[["nr", "top", "bottom", "dz"]].equals(
            result_entire_range[["nr", "top", "bottom", "dz"]]
        )
        assert_array_almost_equal(
            result_single_break["<=0.5"], [1.0, 0.6, 0.5, 1.0, 1.0, 0.2, 0.8, 0.25]
        )
        assert_array_almost_equal(
            result_single_break[">0.5"],
            [np.nan, 0.4, 0.5, np.nan, np.nan, 0.8, 0.2, 0.7],
        )

    @pytest.mark.unittest
    def test_compute_discretized_fractions_layered(
        self, borehole_data, discretization_df
    ):
        subset = borehole_data.gst.select_by_values("nr", ["A", "B"])

        # Discretize every 3 meters
        result = subset.gst.compute_discretized_fractions("lith", 3)
        assert isinstance(result, pd.DataFrame)
        assert_array_equal(result["nr"], ["A", "A", "B", "B"])
        assert_array_equal(result["surface"], [0.2, 0.2, 0.3, 0.3])
        assert_array_equal(result["top"], [0, 3, 0, 3])
        assert_array_equal(result["bottom"], [3, 6, 3, 6])
        assert_array_equal(result["dz"], [3, 3, 3, 3])
        assert_array_almost_equal(result["K"], [0.5, 0.16666667, 0.4, 0.26666667])
        assert_array_almost_equal(result["V"], [np.nan, np.nan, 0.6, 0.03333333])
        assert_array_almost_equal(result["Z"], [0.5, 0.23333333, np.nan, np.nan])

        # Given a standard discretization
        result = subset.gst.compute_discretized_fractions("lith", [0.5, 1.0, 2.0, 4.0])
        assert isinstance(result, pd.DataFrame)
        assert_array_equal(result["nr"], ["A", "A", "A", "A", "B", "B", "B", "B"])
        assert_array_equal(result["surface"], [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3])
        assert_array_almost_equal(
            result["top"], [0.0, 0.5, 1.0, 2.0, 0.0, 0.5, 1.0, 2.0]
        )
        assert_array_almost_equal(
            result["bottom"], [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0]
        )
        assert_array_almost_equal(
            result["dz"], [0.5, 0.5, 1.0, 2.0, 0.5, 0.5, 1.0, 2.0]
        )
        assert_array_almost_equal(
            result["K"], [1.0, 1.0, 0.5, 0.15, 1.0, 1.0, 0.2, 0.4]
        )
        assert_array_almost_equal(
            result["V"], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.8, 0.55]
        )
        assert_array_almost_equal(
            result["Z"], [np.nan, np.nan, 0.5, 0.85, np.nan, np.nan, np.nan, np.nan]
        )

        # Now use all the borehole data, also ones not in the discretization DataFrame, the result should
        # only contain boreholes "A" and "B" because we use `merge_sorted(keep_surveys="inner")`
        result = borehole_data.gst.compute_discretized_fractions(
            "lith", discretization_df[["nr", "bottom"]]
        )
        assert_array_equal(result["nr"], discretization_df["nr"])
        assert_array_equal(result["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(result["bottom"], discretization_df["bottom"])
        assert_array_almost_equal(result["dz"], [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 2.0])
        assert_array_almost_equal(result["K"], [1.0, 1.0, 0.5, 1.0, 1.0, 0.2, 0.4])
        assert_array_almost_equal(
            result["V"], [np.nan, np.nan, np.nan, np.nan, np.nan, 0.8, 0.55]
        )
        assert_array_almost_equal(
            result["Z"], [np.nan, np.nan, 0.5, np.nan, np.nan, np.nan, np.nan]
        )

        # Discretization df with a surface, should apply a surface correction of the layers
        result = borehole_data.gst.compute_discretized_fractions(
            "lith", discretization_df
        )
        assert_array_equal(result["nr"], discretization_df["nr"])
        assert_array_equal(result["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(result["top"], [0.0, 0.3, 0.8, 0.0, 0.6, 1.1, 2.1])
        assert_array_almost_equal(result["bottom"], [0.3, 0.8, 1.8, 0.6, 1.1, 2.1, 4.1])
        assert_array_almost_equal(
            result["surface_correction"], np.repeat([-0.2, 0.1], [3, 4])
        )
        assert_array_almost_equal(result["dz"], [0.3, 0.5, 1.0, 0.6, 0.5, 1.0, 2.0])
        assert_array_almost_equal(result["K"], [1.0, 1.0, 0.7, 1.0, 1.0, 0.1, 0.4])
        assert_array_almost_equal(
            result["V"], [np.nan, np.nan, np.nan, np.nan, np.nan, 0.9, 0.5]
        )
        assert_array_almost_equal(
            result["Z"], [np.nan, np.nan, 0.3, np.nan, np.nan, np.nan, np.nan]
        )

    @pytest.mark.unittest
    def test_compute_discretized_fractions_layered_nap(
        self, borehole_data, discretization_df
    ):
        nap_df = discretization_df.gst._get_depth_relative_to_surface()
        subset = borehole_data.gst.select_by_values("nr", ["A", "B"])

        # Discretize every 5 meters NAP
        result = subset.gst.compute_discretized_fractions(
            "lith", 5, relative_to_reference=True
        )
        assert isinstance(result, pd.DataFrame)
        assert_array_equal(result["nr"], ["A", "A", "B", "B"])
        assert_array_equal(result["surface"], [0.2, 0.2, 0.3, 0.3])
        assert_array_equal(result["top"], [5, 0, 5, 0])
        assert_array_equal(result["bottom"], [0, -5, 0, -5])
        assert_array_equal(result["dz"], [5, 5, 5, 5])
        assert_array_almost_equal(result["K"], [0.04, 0.36, 0.06, 0.34])
        assert_array_almost_equal(result["V"], [np.nan, np.nan, np.nan, 0.38])
        assert_array_almost_equal(result["Z"], [np.nan, 0.44, np.nan, np.nan])

        # Using given NAP boundaries: 1-0, 0--1
        result = subset.gst.compute_discretized_fractions(
            "lith", [1, 0, -1], relative_to_reference=True
        )
        assert_array_equal(result["nr"], ["A", "A", "B", "B"])
        assert_array_equal(result["surface"], [0.2, 0.2, 0.3, 0.3])
        assert_array_almost_equal(result["top"], [1, 0, 1, 0])
        assert_array_almost_equal(result["bottom"], [0, -1, 0, -1])
        assert (result["dz"] == 1).all()
        assert_array_almost_equal(result["K"], [0.2, 1.0, 0.3, 0.9])
        assert_array_almost_equal(result["V"], [np.nan, np.nan, np.nan, 0.1])

        # Now use all the borehole data, also ones not in the discretization DataFrame, the result should
        # only contain boreholes "A" and "B" because we use `merge_sorted(keep_surveys="inner")`
        result = borehole_data.gst.compute_discretized_fractions(
            "lith", nap_df, relative_to_reference=True
        )
        assert_array_equal(result["nr"], nap_df["nr"])
        assert_array_equal(result["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(result["top"], nap_df["top"])
        assert_array_almost_equal(result["bottom"], nap_df["bottom"])
        assert_array_almost_equal(result["dz"], [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 2.0])
        assert_array_almost_equal(result["K"], [0.6, 1.0, 0.7, 1.0, 1.0, 0.1, 0.4])
        assert_array_almost_equal(
            result["V"], [np.nan, np.nan, np.nan, np.nan, np.nan, 0.9, 0.5]
        )
        assert_array_almost_equal(
            result["Z"], [np.nan, np.nan, 0.3, np.nan, np.nan, np.nan, np.nan]
        )

        result = borehole_data.gst.compute_discretized_fractions(
            "lith", nap_df[["nr", "surface", "bottom"]], relative_to_reference=True
        )
        assert_array_equal(result["nr"], nap_df["nr"])
        assert_array_equal(result["surface"], np.repeat([0.2, 0.3], [3, 4]))
        assert_array_almost_equal(result["bottom"], nap_df["bottom"])
        assert_array_almost_equal(result["dz"], [0.3, 0.5, 1.0, 0.6, 0.5, 1.0, 2.0])
        assert_array_almost_equal(result["K"], [1.0, 1.0, 0.7, 1.0, 1.0, 0.1, 0.4])
        assert_array_almost_equal(
            result["V"], [np.nan, np.nan, np.nan, np.nan, np.nan, 0.9, 0.5]
        )
        assert_array_almost_equal(
            result["Z"], [np.nan, np.nan, 0.3, np.nan, np.nan, np.nan, np.nan]
        )

    @pytest.fixture
    def discrete_data(self, cpt_data):
        rng = np.random.default_rng(seed=12)
        cpt_data["nr"] = cpt_data["nr"].str.lower()  # Ensure nr is lowercase
        cpt_data["lith"] = rng.choice(["K", "Z"], size=len(cpt_data))
        return cpt_data

    @pytest.mark.unittest
    def test_compute_discretized_fractions_discrete(
        self, discrete_data, discretization_df
    ):
        discrete_data.gst.compute_discretized_fractions("lith", 5)

    @pytest.mark.unittest
    def test_aggregate_consecutive_layers(self, borehole_data, cpt_data):
        # Combine borehole layers
        result = borehole_data.gst.aggregate_consecutive_layers(
            "lith", keep_original_index=True
        )
        assert isinstance(result, pd.DataFrame)
        assert result.index.equals(
            pd.Index([0, 2, 4, 5, 7, 9, 10, 13, 15, 16, 17, 18, 19, 20])
        )
        assert_array_equal(
            result["lith"],
            ["K", "Z", "K", "K", "V", "K", "K", "Z", "K", "V", "K", "V", "Z", "Z"],
        )
        assert_array_almost_equal(
            result["top"],
            [0.0, 1.5, 3.7, 0.0, 1.2, 3.1, 0.0, 2.9, 0.0, 0.5, 1.2, 1.8, 2.5, 0.0],
        )
        assert_array_almost_equal(
            result["bottom"],
            [1.5, 3.7, 4.2, 1.2, 3.1, 3.9, 2.9, 5.5, 0.5, 1.2, 1.8, 2.5, 3.0, 3.0],
        )

        # Combine borehole layers, aggregate over more than one column.
        borehole_data["categorical_data"] = ["A", "B", "A", "B"] + 21 * ["A"]
        result = borehole_data.gst.aggregate_consecutive_layers(
            ["lith", "categorical_data"], keep_original_index=True
        )
        assert isinstance(result, pd.DataFrame)
        assert result.index.equals(
            pd.Index([0, 1, 2, 3, 4, 5, 7, 9, 10, 13, 15, 16, 17, 18, 19, 20])
        )

        # Combine CPT layers
        cpt_data["categorical_data"] = (
            ["A"] * 4 + ["B"] * 4 + ["A"] * 4 + ["C"] * 4 + ["B"] * 4
        )
        result = cpt_data.gst.aggregate_consecutive_layers(
            "categorical_data", agg_funcs={"qc": "mean", "fs": "max"}
        )
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unittest
    def test_to_pyvista_cylinders(self, borehole_data, cpt_data):
        vtk_object = borehole_data.gst.to_pyvista_cylinders("lith")
        assert isinstance(vtk_object, pv.MultiBlock)

        vtk_object = cpt_data.gst.to_pyvista_cylinders("qc")
        assert isinstance(vtk_object, pv.MultiBlock)

    @pytest.mark.unittest
    def test_to_pyvista_grid(self, borehole_data, cpt_data):
        vtk_object = borehole_data.gst.to_pyvista_grid("lith")
        assert isinstance(vtk_object, pv.UnstructuredGrid)

        vtk_object = cpt_data.gst.to_pyvista_grid("qc")
        assert isinstance(vtk_object, pv.UnstructuredGrid)

    @pytest.mark.unittest
    def test_to_qgis3d(self, borehole_data, tmp_path):
        outfile = tmp_path / r"temp.gpkg"
        borehole_data.gst.to_qgis3d(outfile, crs=28992)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_create_linestrings_3d(self, borehole_data):
        result = borehole_data.gst.create_linestrings_3d()

        assert isinstance(result, gpd.GeoSeries)

        from shapely.geometry import LineString

        assert all(isinstance(geom, LineString) for geom in result)
        assert len(result) == len(borehole_data)

        for geom in result:
            assert len(geom.coords[0]) == 3
            assert len(geom.coords[-1]) == 3

    @pytest.mark.unittest
    def test_to_geopackage_3d(self, borehole_data, tmp_path):
        from shapely import LineString

        outfile = tmp_path / "test_3d.gpkg"
        borehole_data.gst.to_geopackage_3d(outfile)
        assert outfile.is_file()

        test_layers = gpd.list_layers(outfile)
        assert_array_equal(test_layers["name"], ["locations", "3dlines"])
        assert_array_equal(test_layers["geometry_type"], ["Point", "LineString Z"])

        # Check if linestring coordinates are OK
        test_lines = gpd.read_file(outfile, layer="3dlines")
        assert test_lines.geometry.iloc[0].equals(
            LineString([(2, 3, 0.21), (2, 3, -0.6)])
        )
        assert test_lines.geometry.iloc[1].equals(
            LineString([(2, 3, -0.59), (2, 3, -1.3)])
        )

    @pytest.mark.unittest
    def test_to_kingdom(self, borehole_data):
        outfile = Path("temp_kingdom.csv")
        tdfile = Path(outfile.parent, f"{outfile.stem}_TDCHART{outfile.suffix}")
        borehole_data.gst.to_kingdom(outfile)
        assert outfile.is_file()
        assert tdfile.is_file()
        outfile.unlink()
        tdfile.unlink()

    @pytest.mark.unittest
    def test_add_model_data_layered(self, borehole_data, voxelmodel, layermodel):
        """
        Method uses `geost.analysis.combine.add_model_data` which is tested in detail
        in `tests/analysis/test_combine.py`.

        """
        result = borehole_data.gst.add_model_data(
            voxelmodel,
            data_vars="strat",
            aggregate_vars={"lith": "mean"},
            suffix="_model",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (35, 10)
        assert "strat_model" in result.columns
        assert "lith_model" in result.columns

        # Not specifying any data_vars with a layermodel should only add the layermodel's z-dimension
        result = borehole_data.gst.add_model_data(layermodel)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (38, 9)
        assert layermodel.gst.z_dim in result.columns

        result = borehole_data.gst.add_model_data(
            layermodel,
            data_vars="layer",
            aggregate_vars={"kh": "mean"},
            suffix="_model",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (38, 10)
        assert "layer_model" in result.columns
        assert "kh_model" in result.columns

    @pytest.mark.unittest
    def test_add_model_data_discrete(self, cpt_data, voxelmodel, layermodel):
        """
        Method uses `geost.analysis.combine.add_model_data` which is tested in detail
        in `tests/analysis/test_combine.py`.

        """
        result = cpt_data.gst.add_model_data(
            voxelmodel,
            data_vars="strat",
            aggregate_vars={"lith": "mean"},
            suffix="_model",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (24, 11)
        assert "strat_model" in result.columns
        assert "lith_model" in result.columns

        # Not specifying any data_vars with a layermodel should only add the layermodel's z-dimension
        result = cpt_data.gst.add_model_data(layermodel)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (27, 10)
        assert layermodel.gst.z_dim in result.columns

        result = cpt_data.gst.add_model_data(
            layermodel,
            data_vars="layer",
            aggregate_vars={"kh": "mean"},
            suffix="_model",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (27, 11)
        assert "layer_model" in result.columns
        assert "kh_model" in result.columns

    @pytest.fixture
    def data_merge_sorted(self):
        return pd.DataFrame(
            {
                "nr": ["A", "A", "B", "B", "C", "C"],
                "surface": [0.2, 0.2, 0.3, 0.3, 0.25, 0.25],
                "top": [0, 0.8, 0, 1.0, 0, 0.4],
                "bottom": [0.8, 1.2, 1.0, 1.5, 0.4, 0.8],
                "lith": ["K", "K", "Z", "K", "K", "L"],
            }
        )

    @pytest.fixture
    def to_insert_merge_sorted(self):
        return pd.DataFrame(
            {
                "nr": ["A", "B", "D"],
                "surface": [0.3, 0.4, 0.15],
                "top": [0.0, 0.0, 0.0],
                "bottom": [1.1, 2.0, 0.4],
                "value": [15, 35, 45],
            }
        )

    @pytest.mark.unittest
    def test_merge_sorted_outer(self, data_merge_sorted, to_insert_merge_sorted):
        expected_result = pd.DataFrame(
            {
                "nr": ["A", "A", "A", "B", "B", "B", "C", "C", "D"],
                "surface": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.25, 0.25, 0.15],
                "top": [0.0, 0.8, 1.0, 0.0, 1.0, 1.5, 0.0, 0.4, 0.0],
                "bottom": [0.8, 1.0, 1.2, 1.0, 1.5, 1.9, 0.4, 0.8, 0.4],
                "lith": ["K", "K", "K", "Z", "K", np.nan, "K", "L", np.nan],
                "surface_right": [
                    0.3,
                    0.3,
                    np.nan,
                    0.4,
                    0.4,
                    0.4,
                    np.nan,
                    np.nan,
                    0.15,
                ],
                "value": [
                    15.0,
                    15.0,
                    np.nan,
                    35.0,
                    35.0,
                    35.0,
                    np.nan,
                    np.nan,
                    45.0,
                ],
            }
        )

        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted, keep_surveys="outer"
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_result)

        # Drop overlapping columns from the right
        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted, keep_surveys="outer", drop_overlapping_columns=True
        )
        pd.testing.assert_frame_equal(
            result, expected_result.drop(columns="surface_right")
        )

        # Without backfilling the result
        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted, keep_surveys="outer", backfill=False
        )
        assert_array_equal(result["nr"], expected_result["nr"])
        assert_array_equal(result["surface"], expected_result["surface"])
        assert_array_almost_equal(result["top"], expected_result["top"])
        assert_array_almost_equal(result["bottom"], expected_result["bottom"])
        # Test the last columns with `assert_frame_equal` because this automatically deals with Arrow types
        pd.testing.assert_frame_equal(
            result[["lith", "surface_right", "value"]],
            pd.DataFrame(
                {
                    "lith": ["K", np.nan, "K", "Z", "K", np.nan, "K", "L", np.nan],
                    "surface_right": [
                        np.nan,
                        0.3,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.4,
                        np.nan,
                        np.nan,
                        0.15,
                    ],
                    "value": [
                        np.nan,
                        15.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        35.0,
                        np.nan,
                        np.nan,
                        45.0,
                    ],
                }
            ),
        )

        # Test when "other" is relative to a reference such as "NAP"
        nap_insert = to_insert_merge_sorted.gst._get_depth_relative_to_surface()
        result = data_merge_sorted.gst.merge_sorted(
            nap_insert, keep_surveys="outer", relative_to_reference=True
        )
        pd.testing.assert_frame_equal(result, expected_result)

        # Other names of depth columns and surface should work too, if they are accepter positional columns
        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted.rename(
                columns={
                    "nr": "nitg",
                    "bottom": "depth",
                    "surface": "mv",
                }
            ),
            keep_surveys="outer",
        )
        assert_array_equal(result["mv"], expected_result["surface_right"])
        pd.testing.assert_frame_equal(
            result.drop(columns="mv"), expected_result.drop(columns="surface_right")
        )

        with pytest.raises(ValueError, match="Invalid value for keep_surveys"):
            data_merge_sorted.gst.merge_sorted(
                to_insert_merge_sorted, keep_surveys="invalid"
            )

    @pytest.mark.unittest
    def test_merge_sorted_inner(self, data_merge_sorted, to_insert_merge_sorted):
        expected_result = pd.DataFrame(
            {
                "nr": ["A", "A", "A", "B", "B", "B"],
                "surface": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
                "top": [0.0, 0.8, 1.0, 0.0, 1.0, 1.5],
                "bottom": [0.8, 1.0, 1.2, 1.0, 1.5, 1.9],
                "lith": ["K", "K", "K", "Z", "K", np.nan],
                "surface_right": [
                    0.3,
                    0.3,
                    np.nan,
                    0.4,
                    0.4,
                    0.4,
                ],
                "value": [
                    15.0,
                    15.0,
                    np.nan,
                    35.0,
                    35.0,
                    35.0,
                ],
            }
        )

        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted, keep_surveys="inner"
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_result)

    @pytest.mark.unittest
    def test_merge_sorted_left(self, data_merge_sorted, to_insert_merge_sorted):
        expected_result = pd.DataFrame(
            {
                "nr": ["A", "A", "A", "B", "B", "B", "C", "C"],
                "surface": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.25, 0.25],
                "top": [0.0, 0.8, 1.0, 0.0, 1.0, 1.5, 0.0, 0.4],
                "bottom": [0.8, 1.0, 1.2, 1.0, 1.5, 1.9, 0.4, 0.8],
                "lith": ["K", "K", "K", "Z", "K", np.nan, "K", "L"],
                "surface_right": [
                    0.3,
                    0.3,
                    np.nan,
                    0.4,
                    0.4,
                    0.4,
                    np.nan,
                    np.nan,
                ],
                "value": [
                    15.0,
                    15.0,
                    np.nan,
                    35.0,
                    35.0,
                    35.0,
                    np.nan,
                    np.nan,
                ],
            }
        )

        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted, keep_surveys="left"
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_result)

    @pytest.mark.unittest
    def test_merge_sorted_right(self, data_merge_sorted, to_insert_merge_sorted):
        expected_result = pd.DataFrame(
            {
                "nr": ["A", "A", "A", "B", "B", "B", "D"],
                "surface": [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.15],
                "top": [0.0, 0.8, 1.0, 0.0, 1.0, 1.5, 0.0],
                "bottom": [0.8, 1.0, 1.2, 1.0, 1.5, 1.9, 0.4],
                "lith": ["K", "K", "K", "Z", "K", np.nan, np.nan],
                "surface_right": [
                    0.3,
                    0.3,
                    np.nan,
                    0.4,
                    0.4,
                    0.4,
                    0.15,
                ],
                "value": [
                    15.0,
                    15.0,
                    np.nan,
                    35.0,
                    35.0,
                    35.0,
                    45.0,
                ],
            }
        )

        result = data_merge_sorted.gst.merge_sorted(
            to_insert_merge_sorted, keep_surveys="right"
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_result)
