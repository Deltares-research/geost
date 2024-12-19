import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shapely.geometry import LineString

from geost.models import model_utils as mut


@pytest.fixture
def line():
    return LineString([[1, 1], [3, 3]])


@pytest.fixture
def array_2d():
    return np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 20, 10, 20, 10],
            [20, 10, 20, 20, 10],
            [20, 20, 20, 10, 10],
        ]
    )


@pytest.mark.unittest
def test_interpolate_point(line):
    loc, x, y = mut._interpolate_point(line, 1)
    assert loc == 1
    assert np.isclose(x, 1.7071)
    assert np.isclose(y, 1.7071)


@pytest.mark.unittest
def test_sample_along_line(xarray_dataset, line):
    sampled = mut.sample_along_line(xarray_dataset, line, nsamples=3)
    assert sampled.sizes == {"dist": 3, "z": 4}
    assert_array_almost_equal(sampled["dist"], [0.0, 1.414, 2.828], decimal=3)

    sampled = mut.sample_along_line(xarray_dataset, line, dist=0.5)
    assert sampled.sizes == {"dist": 6, "z": 4}
    assert_array_almost_equal(sampled["dist"], [0, 0.5, 1, 1.5, 2, 2.5])


@pytest.mark.unittest
def test_sample_with_coords(xarray_dataset, borehole_collection):
    coords = borehole_collection.header[["x", "y"]].values
    sampled = mut.sample_with_coords(xarray_dataset, coords)
    assert sampled.sizes == {"idx": 4, "z": 4}
    assert_array_equal(sampled["idx"], [0, 1, 2, 4])


@pytest.mark.unittest
def test_label_consecutive_2d(array_2d):
    result = mut.label_consecutive_2d(array_2d, axis=0)
    assert_array_equal(
        result, [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 2, 1, 1, 0], [1, 3, 1, 2, 0]]
    )
    assert result.dtype == array_2d.dtype

    result = mut.label_consecutive_2d(array_2d, axis=1)
    assert_array_equal(
        result, [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 1, 2, 2, 3], [0, 0, 0, 1, 1]]
    )
    assert result.dtype == array_2d.dtype
