import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from geost.bro.bro_utils import coordinates_to_cellcenters, flip_ycoordinates


@pytest.fixture
def dataset():
    coords = np.arange(3)
    return xr.DataArray(
        np.ones((3, 3, 3)),
        coords={"y": coords, "x": coords, "z": [0.5, 1.5, 2.5]},
        dims=("y", "x", "z"),
    )


@pytest.mark.unittest
def test_coordinates_to_cellcenters(dataset: xr.DataArray):
    cellsize, dz = 1, 1
    dataset = coordinates_to_cellcenters(dataset, cellsize, dz)
    assert_array_equal(dataset["x"], [0.5, 1.5, 2.5])
    assert_array_equal(dataset["y"], [0.5, 1.5, 2.5])
    assert_array_equal(dataset["z"], [1, 2, 3])


@pytest.mark.unittest
def test_flip_ycoordinates(dataset: xr.DataArray):
    dataset = flip_ycoordinates(dataset)
    assert_array_equal(dataset["y"], [2, 1, 0])
    assert dataset.dims == ("y", "x", "z")
