from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection

borehole_file = Path(__file__).parent / "data" / "test_boreholes.parquet"
selection_file = Path(__file__).parent / "data" / "test_polygon.parquet"


class TestSpatial:
    @pytest.fixture
    def boreholes(self):
        borehole_collection = read_sst_cores(borehole_file)
        return borehole_collection

    @pytest.fixture
    def selection_polygon(self):
        gdf = gpd.read_parquet(selection_file)
        gdf = gdf.set_crs(epsg=28992)
        return gdf

    @pytest.mark.unittest
    def test_select_within_polygon_no_buffer(self, boreholes, selection_polygon):
        boreholes_selected = boreholes.select_within_polygons(selection_polygon)
        assert boreholes_selected.n_points == 3
        assert len(boreholes_selected.data) == 66

    @pytest.mark.unittest
    def test_select_within_polygon_buffer(self, boreholes, selection_polygon):
        boreholes_selected = boreholes.select_within_polygons(
            selection_polygon, buffer=1000
        )
        assert boreholes_selected.n_points == 8
        assert len(boreholes_selected.data) == 226

    @pytest.mark.unittest
    def test_select_within_polygon_invert(self, boreholes, selection_polygon):
        boreholes_selected = boreholes.select_within_polygons(
            selection_polygon, invert=True
        )
        assert boreholes_selected.n_points == 10
        assert len(boreholes_selected.data) == 284
