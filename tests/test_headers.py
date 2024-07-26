from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from geost.headers import LineHeader, PointHeader


class TestHeaders:
    @pytest.mark.unittest
    def test_init_header(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf)
        assert point_header.__repr__() == "PointHeader instance containing 25 objects"

    @pytest.mark.unittest
    def test_single_get(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf)
        point_header_sel = point_header.get("nr10")
        assert point_header_sel.gdf["nr"].iloc[0] == "nr10"

    @pytest.mark.unittest
    def test_multi_get(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf)
        query = ["nr1", "nr5", "nr10", "nr15", "nr20"]
        point_header_sel = point_header.get(query)
        assert all([nr in point_header_sel.gdf["nr"].values for nr in query])

    @pytest.mark.unittest
    def test_select_within_bbox(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf)
        point_header_sel = point_header.select_within_bbox(1, 3, 1, 3)
        point_header_sel_inverted = point_header.select_within_bbox(
            1, 3, 1, 3, invert=True
        )
        assert len(point_header_sel.gdf) == 9
        assert len(point_header_sel_inverted.gdf) == 16

    @pytest.mark.unittest
    def test_select_with_points(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf)
