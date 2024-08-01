import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.new_base import BoreholeCollection, PointHeader


class TestLayeredData:
    @pytest.mark.unittest
    def test_to_header(self, borehole_data):
        expected_columns = ["nr", "x", "y", "surface", "end", "geometry"]

        header = borehole_data.to_header()

        assert isinstance(header, PointHeader)
        assert_array_equal(header.gdf.columns, expected_columns)
        assert len(header.gdf) == 5
        assert header["nr"].nunique() == 5

    @pytest.mark.unittest
    def test_to_collection(self, borehole_data):
        collection = borehole_data.to_collection()
        assert isinstance(collection, BoreholeCollection)

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_data):
        selected = borehole_data.select_by_values("lith", ["V", "K"], how="or")

        expected_nrs = ["A", "B", "C", "D"]
        selected_nrs = selected["nr"].unique()

        assert_array_equal(selected_nrs, expected_nrs)

        selected = borehole_data.select_by_values("lith", ["V", "K"], how="and")

        expected_nrs = ["B", "D"]
        selected_nrs = selected["nr"].unique()

        assert_array_equal(selected_nrs, expected_nrs)

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_data):
        sliced = borehole_data.slice_by_values("lith", "Z")

        expected_boreholes_with_sand = ["A", "C", "D", "E"]
        expected_length = 10

        assert_array_equal(sliced["nr"].unique(), expected_boreholes_with_sand)
        assert np.all(sliced["lith"] == "Z")
        assert len(sliced) == expected_length

        sliced = borehole_data.slice_by_values("lith", "Z", invert=True)

        expected_boreholes_without_sand = ["A", "B", "C", "D"]
        expected_length = 15

        assert_array_equal(sliced["nr"].unique(), expected_boreholes_without_sand)
        assert ~np.any(sliced["lith"] == "Z")
        assert len(sliced) == expected_length

    @pytest.mark.unittest
    def test_get_cumulative_layer_thickness(self, borehole_data):
        result = borehole_data.get_cumulative_layer_thickness("lith", "V")
        expected_boreholes_returned = ["B", "D"]
        expected_thickness = [1.9, 1.4]

        assert len(result) == 2
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["V"], expected_thickness)

        result = borehole_data.get_cumulative_layer_thickness("lith", ["Z", "K"])
        expected_boreholes_returned = ["A", "B", "C", "D", "E"]
        expected_sand_thickness = [2.2, np.nan, 2.6, 0.5, 3.0]
        expected_clay_thickness = [2.0, 2.0, 2.9, 1.1, np.nan]

        assert result.shape == (5, 2)
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["K"], expected_clay_thickness)
        assert_array_almost_equal(result["Z"], expected_sand_thickness)

    @pytest.mark.unittest
    def test_get_layer_top(self, borehole_data):
        result = borehole_data.get_layer_top("lith", "V")
        expected_boreholes_returned = ["B", "D"]
        expected_tops = [1.2, 0.5]

        assert len(result) == 2
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["V"], expected_tops)

        result = borehole_data.get_layer_top("lith", ["Z", "K"])
        expected_boreholes_returned = ["A", "B", "C", "D", "E"]
        expected_sand_top = [1.5, np.nan, 2.9, 2.5, 0.0]
        expected_clay_top = [0.0, 0.0, 0.0, 0.0, np.nan]

        assert result.shape == (5, 2)
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["Z"], expected_sand_top)
        assert_array_almost_equal(result["K"], expected_clay_top)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, borehole_data):
        upper, lower = 0.6, 2.4
        sliced = borehole_data.slice_depth_interval(upper, lower)

        layers_per_borehole = sliced["nr"].value_counts()
        expected_layer_count = [3, 3, 3, 3, 2]

        assert len(sliced) == 14
        assert sliced["top"].min() == upper
        assert sliced["bottom"].max() == lower
        assert_array_equal(layers_per_borehole, expected_layer_count)

        sliced = borehole_data.slice_depth_interval(
            upper, lower, update_layer_boundaries=False
        )

        expected_tops_of_slice = [0.0, 0.6, 0.0, 0.5, 0.5]
        expected_bottoms_of_slice = [2.5, 2.5, 2.9, 2.5, 2.5]

        tops_of_slice = sliced.df.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.df.groupby("nr")["bottom"].max()

        assert len(sliced) == 14
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        nap_upper, nap_lower = -2, -3
        sliced = borehole_data.slice_depth_interval(
            nap_upper, nap_lower, vertical_reference="NAP"
        )

        expected_tops_of_slice = [2.2, 2.3, 2.25, 2.1, 1.9]
        expected_bottoms_of_slice = [3.2, 3.3, 3.25, 3.0, 2.9]

        tops_of_slice = sliced.df.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.df.groupby("nr")["bottom"].max()

        assert len(sliced) == 11
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        empty_slice = borehole_data.slice_depth_interval(-2, -1)
        empty_slice_nap = borehole_data.slice_depth_interval(
            3, 2, vertical_reference="NAP"
        )

        assert len(empty_slice) == 0
        assert len(empty_slice_nap) == 0

    @pytest.mark.unittest
    def test_to_vtm(self, borehole_data):
        borehole_data.to_vtm()
