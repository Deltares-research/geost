import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shapely.geometry import LineString

from geost.models import model_utils as utils


@pytest.fixture
def line():
    return LineString([[1, 1], [3, 3]])


@pytest.mark.unittest
def test_interpolate_point(line):
    loc, x, y = utils._interpolate_point(line, 1)
    assert loc == 1
    assert np.isclose(x, 1.7071)
    assert np.isclose(y, 1.7071)


@pytest.mark.unittest
def test_sample_along_line(xarray_dataset, line):
    sampled = utils.sample_along_line(xarray_dataset, line, nsamples=3)
    assert sampled.sizes == {"dist": 3, "z": 4}
    assert_array_almost_equal(sampled["dist"], [0.0, 1.414, 2.828], decimal=3)

    sampled = utils.sample_along_line(xarray_dataset, line, dist=0.5)
    assert sampled.sizes == {"dist": 6, "z": 4}
    assert_array_almost_equal(sampled["dist"], [0, 0.5, 1, 1.5, 2, 2.5])


@pytest.mark.unittest
def test_sample_with_coords(xarray_dataset, borehole_collection):
    coords = borehole_collection.header[["x", "y"]].values
    sampled = utils.sample_with_coords(xarray_dataset, coords)
    assert sampled.sizes == {"idx": 4, "z": 4}
    assert_array_equal(sampled["idx"], [0, 1, 2, 4])
