from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import LineString, Point

from geost.new_base import LineHeader, PointHeader


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
        selection_points = [Point(1, 1), Point(4, 4), Point(1, 4), Point(4, 1)]
        selection_gdf = gpd.GeoDataFrame(
            {"id": [0, 1, 2, 3]}, geometry=selection_points
        )
        point_header_sel = point_header.select_with_points(selection_gdf, 1.1)
        point_header_sel_inverted = point_header.select_with_points(
            selection_gdf, 1.1, invert=True
        )
        assert len(point_header_sel.gdf) == 16
        assert len(point_header_sel_inverted.gdf) == 9

    @pytest.mark.unittest
    def test_select_with_lines(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf)
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame({"id": [0, 1]}, geometry=selection_lines)
        point_header_sel = point_header.select_with_lines(selection_gdf, 1)
        point_header_sel_inverted = point_header.select_with_lines(
            selection_gdf, 1, invert=True
        )
        assert len(point_header_sel.gdf) == 21
        assert len(point_header_sel_inverted.gdf) == 4