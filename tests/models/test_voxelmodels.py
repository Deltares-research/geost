import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.models import voxelmodels as vm


@pytest.fixture
def simple_voxelmodel(voxelmodel):
    return voxelmodel.isel(x=[1, 2], y=[1, 2])


@pytest.mark.unittest
def test_slice_depth_interval_values(voxelmodel):
    sliced = vm.slice_depth_interval(voxelmodel, upper=-0.4, lower=-1.6)
    assert isinstance(sliced, xr.Dataset)
    assert sliced.sizes == {"y": 4, "x": 4, "z": 4}
    assert_array_equal(
        sliced["strat"],
        [
            [
                [2.0, 2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, np.nan],
                [1.0, 1.0, 1.0, np.nan],
                [2.0, 1.0, 1.0, np.nan],
            ],
            [
                [2.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, np.nan],
                [1.0, 2.0, 1.0, np.nan],
            ],
            [
                [2.0, 2.0, 1.0, np.nan],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 1.0, np.nan],
            ],
            [
                [2.0, 2.0, 1.0, np.nan],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, np.nan],
                [2.0, 2.0, 1.0, 1.0],
            ],
        ],
    )
    assert_array_equal(
        sliced["lith"],
        (
            [
                [
                    [3.0, 2.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0, np.nan],
                    [1.0, 1.0, 1.0, np.nan],
                    [2.0, 1.0, 1.0, np.nan],
                ],
                [
                    [2.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, np.nan],
                    [3.0, 2.0, 1.0, np.nan],
                ],
                [
                    [3.0, 2.0, 1.0, np.nan],
                    [1.0, 1.0, 3.0, 3.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, np.nan],
                ],
                [
                    [2.0, 2.0, 1.0, np.nan],
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 3.0, np.nan],
                    [2.0, 2.0, 1.0, 1.0],
                ],
            ]
        ),
    )

    # Test with upper and lower bounds at cell boundaries
    sliced = vm.slice_depth_interval(voxelmodel, upper=-0.5, lower=-1.5)
    assert sliced.sizes == {"y": 4, "x": 4, "z": 2}
    assert_array_equal(
        sliced["strat"],
        [
            [[2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            [[2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        ],
    )

    # Test with drop=False --> keep original shape of sliced but set values to NaN
    sliced = vm.slice_depth_interval(voxelmodel, upper=-0.4, lower=-1.6, drop=False)
    assert sliced.sizes == voxelmodel.sizes
    assert_array_equal(
        sliced["strat"],
        [
            [
                [np.nan, 2.0, 2.0, 1.0, 1.0],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
            ],
            [
                [np.nan, 2.0, 1.0, 1.0, 1.0],
                [np.nan, 2.0, 1.0, 1.0, 1.0],
                [np.nan, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 2.0, 1.0, np.nan],
            ],
            [
                [np.nan, 2.0, 2.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0],
                [np.nan, 2.0, 1.0, 1.0, 1.0],
                [np.nan, 2.0, 2.0, 1.0, np.nan],
            ],
            [
                [np.nan, 2.0, 2.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, 1.0],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 2.0, 1.0, 1.0],
            ],
        ],
    )

    # Test when upper is deeper than lower
    sliced = vm.slice_depth_interval(voxelmodel, upper=-1.5, lower=-0.5)
    assert sliced.sizes == {"y": 4, "x": 4, "z": 0}


@pytest.mark.parametrize("as_array", [True, False], ids=["as_array", "as_dataarray"])
def test_slice_depth_interval_with_grids(voxelmodel, depth_grid, as_array):
    if as_array:
        depth_grid = depth_grid.values

    sliced = vm.slice_depth_interval(voxelmodel, upper=depth_grid, lower=depth_grid - 1)
    assert_array_equal(
        sliced["strat"],
        [
            [
                [np.nan, 2.0, 1.0],
                [np.nan, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
            ],
            [
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
            ],
            [
                [2.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
            [
                [2.0, 2.0, np.nan],
                [1.0, 1.0, np.nan],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
        ],
    )
    assert_array_equal(
        sliced["lith"],
        [
            [
                [np.nan, 2.0, 1.0],
                [np.nan, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
            ],
            [
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [3.0, 2.0, 1.0],
            ],
            [
                [3.0, 2.0, 1.0],
                [1.0, 1.0, 3.0],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
            [
                [2.0, 2.0, np.nan],
                [1.0, 1.0, np.nan],
                [2.0, 1.0, 3.0],
                [2.0, 2.0, 1.0],
            ],
        ],
    )

    # Test with drop=False --> keep original shape of sliced but set values to NaN
    sliced = vm.slice_depth_interval(voxelmodel, depth_grid, depth_grid - 1, drop=False)
    assert sliced.sizes == voxelmodel.sizes
    assert_array_equal(
        sliced["strat"],
        [
            [
                [np.nan, np.nan, 2.0, 1.0, np.nan],
                [np.nan, np.nan, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
            ],
            [
                [np.nan, 2.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 1.0, 2.0, 1.0, np.nan],
            ],
            [
                [np.nan, 2.0, 2.0, 1.0, np.nan],
                [np.nan, 1.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 2.0, 1.0, np.nan],
            ],
            [
                [np.nan, 2.0, 2.0, np.nan, np.nan],
                [np.nan, 1.0, 1.0, np.nan, np.nan],
                [np.nan, 2.0, 1.0, 1.0, np.nan],
                [np.nan, 2.0, 2.0, 1.0, np.nan],
            ],
        ],
    )


@pytest.mark.unittest
def test_slice_depth_interval_with_1d_dataarray(voxelmodel):
    da_1d = xr.DataArray([-1.0, -1.5, -2.0, -2.5], dims=["x"])
    sliced = vm.slice_depth_interval(voxelmodel, upper=da_1d, lower=da_1d - 1)
    assert_array_equal(
        sliced["strat"],
        [
            [[np.nan, 2.0, 2.0], [2.0, 2.0, np.nan], [2.0, np.nan, np.nan]],
            [[np.nan, 2.0, 1.0], [2.0, 2.0, np.nan], [2.0, np.nan, np.nan]],
            [[np.nan, 2.0, 2.0], [2.0, 1.0, np.nan], [2.0, np.nan, np.nan]],
            [[np.nan, 2.0, 2.0], [2.0, 1.0, np.nan], [2.0, np.nan, np.nan]],
        ],
    )

    # With 1D Numpy array we cannot broadcast to the dataset dimensions because of unnamed dimensions
    with pytest.raises(
        ValueError, match="Failed to broadcast input array to dataset dimensions"
    ):
        da_1d_invalid = da_1d.values
        vm.slice_depth_interval(
            voxelmodel, upper=da_1d_invalid, lower=da_1d_invalid - 1
        )


@pytest.mark.parametrize(
    "how, upper, lower, result_shape, result_z",
    [
        ("overlap", -0.8, -1.8, (2, 2, 3), [-1.75, -1.25, -0.75]),
        ("overlap", -0.5, -2.0, (2, 2, 3), [-1.75, -1.25, -0.75]),
        ("majority", -0.8, -1.8, (2, 2, 2), [-1.75, -1.25]),
        ("majority", -1.25, -1.75, (2, 2, 2), [-1.75, -1.25]),
        ("inner", -0.8, -1.8, (2, 2, 1), [-1.25]),
        ("inner", -1.0, -1.5, (2, 2, 1), [-1.25]),
        ("invalid", None, None, None, None),
    ],
    ids=[
        "overlap",
        "overlap_exact",
        "majority",
        "majority_exact",
        "inner",
        "inner_exact",
        "invalid",
    ],
)
def test_slice_depth_interval_how(
    simple_voxelmodel, how, upper, lower, result_shape, result_z
):
    if how == "invalid":
        with pytest.raises(ValueError, match="Invalid value for 'how'"):
            vm.slice_depth_interval(
                simple_voxelmodel, upper=upper, lower=lower, how=how
            )
    else:
        sliced = vm.slice_depth_interval(
            simple_voxelmodel, upper=upper, lower=lower, how=how
        )
        assert isinstance(sliced, xr.Dataset)
        assert sliced.gst.shape == result_shape
        assert_array_almost_equal(sliced["z"], result_z)
