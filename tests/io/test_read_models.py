import dask
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

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
        voxelmodel_netcdf,
        data_vars=["strat", "lith"],
        chunks={"x": 2, "y": 2, "z": -1},
        load=False,
    )
    assert isinstance(model, xr.Dataset)
    assert_array_equal(model.data_vars, ["strat", "lith"])
    assert model.chunks == {"y": (2, 2), "x": (2, 2), "z": (5,)}
    # Specify chunks and load=False should load data lazily as dask arrays
    assert all(isinstance(v.data, dask.array.Array) for v in model.data_vars.values())

    model = geost.read_model_netcdf(
        voxelmodel_netcdf, chunks={"x": 2, "y": 2, "z": -1}, load=True
    )  # Load=True should load data into memory as numpy arrays, ignoring chunks
    assert isinstance(model, xr.Dataset)
    assert all(isinstance(v.data, np.ndarray) for v in model.data_vars.values())


@pytest.mark.unittest
def test_read_geotop_netcdf(geotop_netcdf):
    gtp = geost.read_geotop_netcdf(geotop_netcdf)
    assert isinstance(gtp, xr.Dataset)
    assert gtp.sizes == {"x": 5, "y": 5, "z": 101}
    assert gtp.gst.crs == 28992
    assert gtp.gst.bounds() == (110000.0, 440000.0, 110500.0, 440500.0)
    assert_array_almost_equal(gtp["x"], [110050, 110150, 110250, 110350, 110450])
    assert_array_almost_equal(gtp["y"], [440050, 440150, 440250, 440350, 440450])
    assert_array_almost_equal(gtp["z"], np.linspace(-49.75, 0.25, gtp.sizes["z"]))

    bbox = (110_200, 440_200, 110_400, 440_400)
    gtp = geost.read_geotop_netcdf(
        geotop_netcdf,
        data_vars=["strat", "lithok"],
        bbox=bbox,
    )
    assert isinstance(gtp, xr.Dataset)
    assert gtp.sizes == {"x": 2, "y": 2, "z": 101}
    assert_array_equal(gtp.data_vars, ["strat", "lithok"])
    assert gtp.gst.bounds() == bbox
    assert_array_almost_equal(gtp["x"], [110250, 110350])
    assert_array_almost_equal(gtp["y"], [440250, 440350])
    assert_array_almost_equal(gtp["z"], np.linspace(-49.75, 0.25, gtp.sizes["z"]))


@pytest.mark.unittest
def test_read_geotop_from_opendap(geotop_netcdf):
    bbox = (110_200, 440_200, 110_400, 440_400)
    gtp = geost.read_geotop_from_opendap(data_vars=["strat", "lithok"], bbox=bbox)
    assert isinstance(gtp, xr.Dataset)
    assert gtp.gst.crs == 28992
    assert gtp.sizes == {"x": 2, "y": 2, "z": 313}
    assert_array_equal(gtp.data_vars, ["strat", "lithok"])
    assert gtp.gst.bounds() == bbox
    assert_array_almost_equal(gtp["x"], [110250, 110350])
    assert_array_almost_equal(gtp["y"], [440250, 440350])
    assert_array_almost_equal(gtp["z"], np.linspace(-49.75, 106.25, gtp.sizes["z"]))

    # Check if reading from the OpenDAP server gives the same result as reading from the local netcdf file
    gtp_local = geost.read_geotop_netcdf(
        geotop_netcdf, data_vars=["strat", "lithok"], bbox=bbox
    )
    assert gtp.sel(z=gtp_local["z"]).equals(gtp_local)
    # We need to select the same z coordinates because the geotop_netcdf fixture has a subset
    # of the z coordinates to keep the file size of the netcdf file the tests/data directory
    # small.


@pytest.mark.unittest
def test_read_regis_netcdf(regis_netcdf):
    regis = geost.read_regis_netcdf(regis_netcdf)
    assert isinstance(regis, xr.Dataset)
    assert "mv" not in regis["layer"]
    assert_array_equal(
        regis.data_vars, ["top", "bottom", "hgv", "kD", "c", "kh", "kv", "sdh", "sdv"]
    )
    assert_array_equal(regis.coords, ["crs", "x", "y", "layer"])
    assert regis.sizes == {"layer": 131, "y": 5, "x": 5}
    assert regis.gst.crs == 28992
    assert regis.gst.bounds() == (110000.0, 440000.0, 110500.0, 440500.0)
    assert_array_almost_equal(regis["x"], [110050, 110150, 110250, 110350, 110450])
    assert_array_almost_equal(regis["y"], [440050, 440150, 440250, 440350, 440450])

    bbox = (110_200, 440_200, 110_400, 440_400)
    regis = geost.read_regis_netcdf(regis_netcdf, bbox=bbox)
    assert isinstance(regis, xr.Dataset)
    assert regis.sizes == {"layer": 131, "y": 2, "x": 2}
    assert regis.gst.bounds() == bbox
    assert_array_almost_equal(regis["x"], [110250, 110350])
    assert_array_almost_equal(regis["y"], [440250, 440350])

    # Test usage of data_vars behavior
    regis = geost.read_regis_netcdf(regis_netcdf, data_vars=["hgv", "kD"])
    # Not including top and bottom in data_vars should still include them in the result
    # otherwise, the read data is not a valid layermodel and the .gst accessor will not work.
    assert_array_equal(regis.data_vars, ["top", "bottom", "hgv", "kD"])

    regis = geost.read_regis_netcdf(
        regis_netcdf, data_vars=["top", "bottom", "hgv", "kD"]
    )  # Adding top and bottom to data_vars should off course include them in the result
    assert_array_equal(regis.data_vars, ["top", "bottom", "hgv", "kD"])


@pytest.mark.unittest
def test_read_regis_from_opendap(regis_netcdf):
    bbox = (110_200, 440_200, 110_400, 440_400)
    regis = geost.read_regis_from_opendap(data_vars=["hgv", "kD"], bbox=bbox)
    assert isinstance(regis, xr.Dataset)
    assert "mv" not in regis["layer"]
    assert_array_equal(regis.data_vars, ["top", "bottom", "hgv", "kD"])
    assert_array_equal(regis.coords, ["crs", "x", "y", "layer"])
    assert regis.sizes == {"layer": 131, "y": 2, "x": 2}
    assert regis.gst.crs == 28992
    assert regis.gst.bounds() == bbox
    assert_array_almost_equal(regis["x"], [110250, 110350])
    assert_array_almost_equal(regis["y"], [440250, 440350])

    # Check if reading from the OpenDAP server gives the same result as reading from the local netcdf file
    regis_local = geost.read_regis_netcdf(
        regis_netcdf, data_vars=["hgv", "kD"], bbox=bbox
    )
    assert regis.equals(regis_local)
