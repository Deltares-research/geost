import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import geost


@pytest.fixture
def voxelmodel_netcdf(tmp_path, voxelmodel):
    nc_path = tmp_path / "voxelmodel.nc"
    voxelmodel.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def layermodel_netcdf(tmp_path, layermodel):
    nc_path = tmp_path / "layermodel.nc"
    layermodel.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def geotop_netcdf(testdatadir):
    return testdatadir / "geotop_small_selection.nc"


@pytest.fixture
def regis_netcdf(testdatadir):
    return testdatadir / "regis_small_selection.nc"


@pytest.mark.unittest
def test_read_model_netcdf(voxelmodel_netcdf):
    model = geost.read_model_netcdf(voxelmodel_netcdf)
    assert isinstance(model, xr.Dataset)

    model = geost.read_model_netcdf(
        voxelmodel_netcdf, data_vars="strat", bbox=(1, 1, 3, 3), load=True
    )
    assert isinstance(model, xr.DataArray)
    assert model.gst.bounds() == (1, 1, 3, 3)
    assert model.name == "strat"

    model = geost.read_model_netcdf(
        voxelmodel_netcdf, data_vars=["strat", "lith"], chunks={"x": 2, "y": 2, "z": -1}
    )
    assert isinstance(model, xr.Dataset)
    assert_array_equal(model.data_vars, ["strat", "lith"])
    assert model.chunks == {"y": (2, 2), "x": (2, 2), "z": (5,)}
