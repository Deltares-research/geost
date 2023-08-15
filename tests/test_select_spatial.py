import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pathlib import Path

from pysst.borehole import BoreholeCollection
from pysst import read_sst_cores

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
        return gdf

    @pytest.mark.unittes
    def test_select_within_polygon_no_buffer(self, boreholes, selection_polygon):
        boreholes_selected = boreholes.select_within_polygons(selection_polygon)
        assert boreholes_selected.n_points == 3

    @pytest.mark.unittest
    def test_select_within_polygon_buffer(self, boreholes, selection_polygon):
        boreholes_selected = boreholes.select_within_polygons(
            selection_polygon, buffer=1000
        )
        assert boreholes_selected.n_points == 8

    @pytest.mark.unittest
    def test_select_within_polygon_invert(self, boreholes, selection_polygon):
        boreholes_selected = boreholes.select_within_polygons(
            selection_polygon, invert=True
        )
        assert boreholes_selected.n_points == 10
