import pytest
from numpy.testing import assert_array_equal

from geost.new_base import PointHeader


class TestLayeredData:
    @pytest.mark.unittest
    def test_to_header(self, borehole_data):
        expected_columns = ["nr", "x", "y", "mv", "end", "geometry"]

        header = borehole_data.to_header()

        assert isinstance(header, PointHeader)
        assert_array_equal(header.gdf.columns, expected_columns)
        assert len(header.gdf) == 13
        assert header.gdf["nr"].nunique() == 13
