import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shapely.geometry import LineString, Polygon

from geost.accessor import GeostFrame
from geost.accessors.accessor import DATA_BACKEND, HEADER_BACKEND
from tests.conftest import borehole_data


@pytest.fixture
def dataframe():
    return pd.DataFrame({"nr": ["a", "b"], "x": [1, 2], "y": [3, 4]})


@pytest.fixture
def geodataframe(dataframe):
    return gpd.GeoDataFrame(
        dataframe, geometry=gpd.points_from_xy(dataframe.x, dataframe.y)
    )


class TestGeostFrame:
    @pytest.mark.unittest
    def test_accessor(self, dataframe: pd.DataFrame):
        assert hasattr(dataframe, "gst")
        assert isinstance(dataframe.gst, GeostFrame)
        assert dataframe.gst._top is None
        assert dataframe.gst._bottom is None

        with pytest.raises(
            KeyError,
            match="DataFrame must contain an 'nr' column identifying individual objects.",
        ):
            df_invalid = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            df_invalid.gst  # Should raise an error because 'nr' column is missing

    @pytest.mark.parametrize(
        "df, top, bottom",
        [
            (pd.DataFrame(columns=["nr", "top", "bottom"]), "top", "bottom"),
            (
                pd.DataFrame(columns=["nr", "top", "bottom", "depth"]),
                "top",
                "depth",
            ),  # 'depth' should take precedence over 'bottom'
            (pd.DataFrame(columns=["nr", "bottom"]), None, "bottom"),
            (pd.DataFrame(columns=["nr", "depth"]), None, "depth"),
            (pd.DataFrame(columns=["nr"]), None, None),  # No depth columns present
        ],
    )
    def test_set_depth_columns(self, df, top, bottom):
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
        with pytest.raises(
            TypeError,
            match="Object is not a GeoDataFrame with a valid geometry column.",
        ):
            dataframe.gst._check_has_geometry()

        assert geodataframe.gst.has_geometry
        geodataframe.gst._check_has_geometry()  # Should not raise an error for geodataframe

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
            df.gst._check_has_depth()  # Should not raise an error for valid depth column combinations
        else:
            assert not df.gst.has_depth_columns
            with pytest.raises(KeyError):
                df.gst._check_has_depth()

    @pytest.mark.unittest
    def test_to_iterable(self, borehole_data):
        inst = "string"
        inst = borehole_data.gst._to_iterable(inst)
        assert isinstance(inst, list)

        inst = ["list of strings"]
        inst = borehole_data.gst._to_iterable(inst)
        assert isinstance(inst, list)

    @pytest.mark.unittest
    def test_select_within_bbox(self, point_header):
        selected = point_header.gst.select_within_bbox(1, 1, 3, 3)
        assert len(selected) == 9

        # Test inverted selection: everything outside the bbox
        selected = point_header.gst.select_within_bbox(1, 1, 3, 3, invert=True)
        assert len(selected) == 16

        with pytest.raises(TypeError):
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

        with pytest.raises(TypeError):
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

        with pytest.raises(TypeError):
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

        with pytest.raises(TypeError):
            point_header = point_header.drop(columns="geometry")
            # Remove geometry column to make it invalid for spatial selection
            point_header.gst.select_within_polygons(selection_gdf)

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

        result = borehole_data.gst.get_layer_top("lith", ["Z", "V"], min_thickness=1.0)
        assert_array_equal(result.index, ["A", "B", "C"])
        assert_array_almost_equal(result, [1.5, 1.2, 3.8])

        result = borehole_data.gst.get_layer_top("bottom", slice(1.5, 3.1))
        assert_array_equal(result.index, ["A", "B", "C", "D", "E"])
        assert_array_almost_equal(result, [0.8, 1.2, 1.4, 1.2, 1.2])

    @pytest.mark.unittest
    def test_get_layer_top_discrete(self, cpt_data):
        result = cpt_data.gst.get_layer_top("qc", slice(0.7, 18))
        assert (
            "thickness" not in cpt_data.columns
        )  # Ensure thickness column is not added to original DataFrame
        assert_array_equal(result.index, ["a", "b"])
        assert_array_equal(result, [8.0, 0.0])

        result = cpt_data.gst.get_layer_top("qc", slice(0.7, 18), min_thickness=2.0)


class TestHeaderAccessor:
    @pytest.fixture
    def point(self):
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy([0, 1], [0, 1]))

    @pytest.fixture
    def linestring(self):
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(
                ["LINESTRING (0 0, 1 1)", "LINESTRING (1 0, 0 1)"]
            )
        )

    @pytest.fixture
    def polygon(self):
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                    "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
                ]
            )
        )

    @pytest.fixture
    def invalid(self):
        return pd.DataFrame()  # No geometry column is invalid for the accessor.

    @pytest.mark.unittest
    def test_get_geom_type(self, invalid):
        with pytest.raises(
            TypeError, match="Header accessor only accepts GeoDataFrames."
        ):
            invalid.gsthd

    @pytest.mark.parametrize(
        "header",
        ["point", "linestring", "polygon"],
        ids=["point", "linestring", "polygon"],
    )
    def test_backend_selection(self, header, request):
        """
        Test to check whether the correct backend is selected based on the 'headertype'
        attribute.

        """
        if header == "polygon":
            with pytest.raises(
                TypeError, match="No Header backend available for polygon"
            ):
                request.getfixturevalue(header).gsthd
        else:
            backend = HEADER_BACKEND[header]
            header = request.getfixturevalue(header)
            assert isinstance(header.gsthd._backend, backend)


class TestDataAccessor:
    @pytest.fixture
    def layered(self):
        df = pd.DataFrame({"top": [0, 30], "bottom": [30, 40]})
        return df

    @pytest.fixture
    def discrete(self):
        df = pd.DataFrame({"depth": [1, 2, 3]})
        return df

    @pytest.fixture
    def invalid(self):
        return (
            pd.DataFrame()
        )  # No "top", "bottom", or "depth" columns are invalid for the accessor.

    @pytest.mark.parametrize(
        "datatype",
        ["layered", "discrete", "invalid"],
        ids=["layered", "discrete", "invalid"],
    )
    def test_backend_selection(self, datatype, request):
        """
        Test to check whether the correct backend is selected based on the 'datatype'
        attribute.

        """
        df = request.getfixturevalue(datatype)
        if datatype == "invalid":
            expected_error = (
                "No 'top' and 'bottom' or 'depth' columns present. Data accessor cannot "
                "determine 'layered' or 'discrete' backend."
            )
            with pytest.raises(KeyError, match=expected_error):
                df.gstda
        else:
            backend = DATA_BACKEND[datatype]
            assert isinstance(df.gstda._backend, backend)
