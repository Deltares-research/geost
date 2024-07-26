import pytest
from numpy.testing import assert_array_equal

from geost.new_base import BoreholeCollection, PointHeader


class TestLayeredData:
    @pytest.mark.unittest
    def test_to_header(
        self, borehole_data
    ):  # TODO: Use __getitem__ functionality of header
        expected_columns = ["nr", "x", "y", "mv", "end", "geometry"]

        header = borehole_data.to_header()

        assert isinstance(header, PointHeader)
        assert_array_equal(header.gdf.columns, expected_columns)
        assert len(header.gdf) == 5
        assert header.gdf["nr"].nunique() == 5

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
