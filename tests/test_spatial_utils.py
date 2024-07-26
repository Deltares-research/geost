from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from geost import read_sst_cores, spatial
from geost.borehole import BoreholeCollection
from geost.utils import dataframe_to_geodataframe


class TestSpatialUtils:
    @pytest.fixture
    def raster(self):
        x_coors = np.arange(1, 4)
        y_coors = np.arange(4, 1, -1)
        data = np.ones((3, 3))
        array = xr.DataArray(data, {"x": x_coors, "y": y_coors})
        return array

    @pytest.fixture
    def invalid_raster(self):
        x_coors = np.arange(1, 4)
        y_coors = np.arange(4, 1, -1)
        data = np.ones((3, 3))
        array = xr.DataArray(data, {"invalid_x": x_coors, "invalid_y": y_coors})
        return array

    @pytest.fixture
    def dataframe_with_coordinates(self):
        dataframe = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        return dataframe

    @pytest.mark.unittest
    def test_dataframe_to_geodataframe(self, dataframe_with_coordinates):
        gdf = dataframe_to_geodataframe(dataframe_with_coordinates, crs=28992)
        assert isinstance(gdf["geometry"].dtype, gpd.array.GeometryDtype)

    @pytest.mark.unittest
    def test_get_raster_values(self, raster, dataframe_with_coordinates):
        x = dataframe_with_coordinates["x"].values
        y = dataframe_with_coordinates["y"].values
        raster_values = spatial.get_raster_values(x, y, raster)
        assert_allclose(raster_values, np.array([np.nan, 1.0, 1.0, np.nan, np.nan]))

    @pytest.mark.unittest
    def test_get_invalid_raster_values(
        self, invalid_raster, dataframe_with_coordinates
    ):
        # In this case the used raster does not have the correct labels for x and y
        # coordinates, which should raise an error.
        x = dataframe_with_coordinates["x"].values
        y = dataframe_with_coordinates["y"].values
        with pytest.raises(Exception) as error_info:
            spatial.get_raster_values(x, y, invalid_raster)
        assert error_info.errisinstance(TypeError)
        assert error_info.match(
            "The xr.DataArray to sample from does not have the "
            + "required 'x' and 'y' dimensions"
        )
