from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from geost import read_sst_cores
from geost.borehole import BoreholeCollection

borehole_file = Path(__file__).parent / "data" / "test_boreholes.parquet"
selection_file = Path(__file__).parent / "data" / "test_polygon.parquet"


class TestPointCollection:

    @pytest.fixture
    def boreholes(self):
        borehole_collection = read_sst_cores(self.borehole_file)
        return borehole_collection

    @pytest.mark.unittest
    def test_get_raster_values(self):
        pass
