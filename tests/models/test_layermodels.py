import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.models import layermodels


@pytest.mark.unittest
def test_slice_depth_interval_values(layermodel):
    sliced = layermodels.slice_depth_interval(layermodel, upper=-0.4, lower=-1.6)
    assert isinstance(sliced, xr.Dataset)
    assert sliced.sizes == {"y": 4, "x": 4, "layer": 3}
    assert_array_equal(sliced.data_vars, layermodel.data_vars)
    assert_array_equal(sliced["layer"], ["B", "C", "D"])
    assert_array_equal(sliced["surface"], layermodel["surface"])  # Should be unchanged
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [-0.4, np.nan, -1.05],
                [-0.4, np.nan, -0.85],
                [-0.4, -0.95, np.nan],
                [np.nan, -0.4, np.nan],
            ],
            [
                [-0.4, np.nan, -1.05],
                [-0.4, np.nan, -0.85],
                [-0.4, -0.95, np.nan],
                [np.nan, -0.4, np.nan],
            ],
            [
                [-0.4, np.nan, -1.05],
                [-0.4, -0.85, np.nan],
                [np.nan, -0.4, np.nan],
                [np.nan, np.nan, -0.4],
            ],
            [
                [np.nan, np.nan, -0.4],
                [np.nan, -0.4, np.nan],
                [np.nan, -0.4, np.nan],
                [np.nan, np.nan, -0.4],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [-1.05, np.nan, -1.6],
                [-0.85, np.nan, -1.6],
                [-0.95, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
            ],
            [
                [-1.05, np.nan, -1.6],
                [-0.85, np.nan, -1.6],
                [-0.95, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
            ],
            [
                [-1.05, np.nan, -1.6],
                [-0.85, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
                [np.nan, np.nan, -1.6],
            ],
            [
                [np.nan, np.nan, -1.6],
                [np.nan, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
                [np.nan, np.nan, -1.6],
            ],
        ],
    )
    # Check that the other variables are also sliced correctly
    assert_array_equal(sliced["top"].notnull(), sliced["thickness"].notnull())
    assert_array_equal(sliced["top"].notnull(), sliced["kh"].notnull())

    # Test update_top_bottom=False and drop=False --> keeps original shape and values
    sliced = layermodel.gst.slice_depth_interval(
        upper=-0.4, lower=-1.6, update_top_bottom=False, drop=False
    )
    assert isinstance(sliced, xr.Dataset)
    assert sliced.sizes == {"y": 4, "x": 4, "layer": 4}
    assert_array_equal(sliced["layer"], layermodel["layer"])
    assert_array_equal(sliced["surface"], layermodel["surface"])  # Should be unchanged
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, np.nan, -0.85],
                [np.nan, -0.2, -0.95, np.nan],
                [np.nan, np.nan, -0.35, np.nan],
            ],
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, np.nan, -0.85],
                [np.nan, -0.2, -0.95, np.nan],
                [np.nan, np.nan, -0.35, np.nan],
            ],
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, -0.85, np.nan],
                [np.nan, np.nan, -0.2, np.nan],
                [np.nan, np.nan, np.nan, -0.35],
            ],
            [
                [np.nan, np.nan, np.nan, -0.25],
                [np.nan, np.nan, -0.15, np.nan],
                [np.nan, np.nan, -0.2, np.nan],
                [np.nan, np.nan, np.nan, -0.35],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, np.nan, -3.25],
                [np.nan, -0.95, -2.55, np.nan],
                [np.nan, np.nan, -2.15, np.nan],
            ],
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, np.nan, -3.25],
                [np.nan, -0.95, -2.55, np.nan],
                [np.nan, np.nan, -2.15, np.nan],
            ],
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, -2.45, np.nan],
                [np.nan, np.nan, -2.0, np.nan],
                [np.nan, np.nan, np.nan, -2.95],
            ],
            [
                [np.nan, np.nan, np.nan, -3.15],
                [np.nan, np.nan, -1.95, np.nan],
                [np.nan, np.nan, -2.0, np.nan],
                [np.nan, np.nan, np.nan, -2.95],
            ],
        ],
    )
    # Check that the other variables are also sliced correctly
    assert_array_equal(sliced["top"].notnull(), sliced["thickness"].notnull())
    assert_array_equal(sliced["top"].notnull(), sliced["kh"].notnull())

    # Test with only upper bound
    sliced = layermodels.slice_depth_interval(layermodel, upper=-0.4)
    assert_array_equal(sliced["layer"], ["B", "C", "D"])
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [-0.4, np.nan, -1.05],
                [-0.4, np.nan, -0.85],
                [-0.4, -0.95, -2.55],
                [np.nan, -0.4, -2.15],
            ],
            [
                [-0.4, np.nan, -1.05],
                [-0.4, np.nan, -0.85],
                [-0.4, -0.95, -2.55],
                [np.nan, -0.4, -2.15],
            ],
            [
                [-0.4, np.nan, -1.05],
                [-0.4, -0.85, -2.45],
                [np.nan, -0.4, -2.0],
                [np.nan, np.nan, -0.4],
            ],
            [
                [np.nan, np.nan, -0.4],
                [np.nan, -0.4, -1.95],
                [np.nan, -0.4, -2.0],
                [np.nan, np.nan, -0.4],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [-1.05, np.nan, -3.25],
                [-0.85, np.nan, -3.25],
                [-0.95, -2.55, -3.35],
                [np.nan, -2.15, -3.35],
            ],
            [
                [-1.05, np.nan, -3.25],
                [-0.85, np.nan, -3.25],
                [-0.95, -2.55, -3.35],
                [np.nan, -2.15, -3.35],
            ],
            [
                [-1.05, np.nan, -3.25],
                [-0.85, -2.45, -3.25],
                [np.nan, -2.0, -3.2],
                [np.nan, np.nan, -2.95],
            ],
            [
                [np.nan, np.nan, -3.15],
                [np.nan, -1.95, -3.15],
                [np.nan, -2.0, -3.2],
                [np.nan, np.nan, -2.95],
            ],
        ],
    )

    # Test with only lower bound
    sliced = layermodels.slice_depth_interval(layermodel, lower=0)
    assert_array_equal(sliced["layer"], ["A"])
    assert_array_almost_equal(
        sliced["top"],
        [
            [[0.2], [0.3], [0.25], [0.1]],
            [[0.2], [0.3], [0.25], [0.1]],
            [[0.2], [0.3], [0.25], [0.1]],
            [[0.2], [0.3], [0.25], [0.1]],
        ],
    )
    assert (sliced["bottom"] == 0).all()


@pytest.mark.parametrize("as_array", [True, False], ids=["as_array", "as_dataarray"])
def test_slice_depth_interval_grid(layermodel, depth_grid, as_array):
    if as_array:
        depth_grid = depth_grid.values

    sliced = layermodels.slice_depth_interval(
        layermodel, upper=depth_grid, lower=depth_grid - 1
    )
    assert isinstance(sliced, xr.Dataset)
    assert sliced.sizes == {"y": 4, "x": 4, "layer": 3}
    assert_array_equal(sliced.data_vars, layermodel.data_vars)
    assert_array_equal(sliced["layer"], ["B", "C", "D"])
    assert_array_equal(sliced["surface"], layermodel["surface"])  # Should be unchanged
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [-0.5, np.nan, -1.05],
                [-0.5, np.nan, -0.85],
                [-0.8, -0.95, np.nan],
                [np.nan, -0.7, np.nan],
            ],
            [
                [-0.8, np.nan, -1.05],
                [-0.8, np.nan, -0.85],
                [-0.8, -0.95, np.nan],
                [np.nan, -0.8, np.nan],
            ],
            [
                [-0.7, np.nan, -1.05],
                [-0.7, -0.85, np.nan],
                [np.nan, -0.7, np.nan],
                [np.nan, np.nan, -0.7],
            ],
            [
                [np.nan, np.nan, -1.0],
                [np.nan, -1.0, -1.95],
                [np.nan, -0.6, np.nan],
                [np.nan, np.nan, -0.6],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [-1.05, np.nan, -1.5],
                [-0.85, np.nan, -1.5],
                [-0.95, -1.8, np.nan],
                [np.nan, -1.7, np.nan],
            ],
            [
                [-1.05, np.nan, -1.8],
                [-0.85, np.nan, -1.8],
                [-0.95, -1.8, np.nan],
                [np.nan, -1.8, np.nan],
            ],
            [
                [-1.05, np.nan, -1.7],
                [-0.85, -1.7, np.nan],
                [np.nan, -1.7, np.nan],
                [np.nan, np.nan, -1.7],
            ],
            [
                [np.nan, np.nan, -2.0],
                [np.nan, -1.95, -2.0],
                [np.nan, -1.6, np.nan],
                [np.nan, np.nan, -1.6],
            ],
        ],
    )
    assert (sliced["top"].max(dim="layer") <= depth_grid).all()
    assert (sliced["bottom"].min(dim="layer") >= depth_grid - 1).all()
    # Check that the other variables are also sliced correctly
    assert_array_equal(sliced["top"].notnull(), sliced["thickness"].notnull())
    assert_array_equal(sliced["top"].notnull(), sliced["kh"].notnull())

    # Test update_top_bottom=False and drop=False --> keeps original shape and values
    sliced = layermodel.gst.slice_depth_interval(
        upper=depth_grid, lower=depth_grid - 1, update_top_bottom=False, drop=False
    )
    assert sliced.sizes == layermodel.sizes
    assert_array_equal(sliced["layer"], layermodel["layer"])
    assert_array_equal(sliced["surface"], layermodel["surface"])
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, np.nan, -0.85],
                [np.nan, -0.2, -0.95, np.nan],
                [np.nan, np.nan, -0.35, np.nan],
            ],
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, np.nan, -0.85],
                [np.nan, -0.2, -0.95, np.nan],
                [np.nan, np.nan, -0.35, np.nan],
            ],
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, -0.85, np.nan],
                [np.nan, np.nan, -0.2, np.nan],
                [np.nan, np.nan, np.nan, -0.35],
            ],
            [
                [np.nan, np.nan, np.nan, -0.25],
                [np.nan, np.nan, -0.15, -1.95],
                [np.nan, np.nan, -0.2, np.nan],
                [np.nan, np.nan, np.nan, -0.35],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, np.nan, -3.25],
                [np.nan, -0.95, -2.55, np.nan],
                [np.nan, np.nan, -2.15, np.nan],
            ],
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, np.nan, -3.25],
                [np.nan, -0.95, -2.55, np.nan],
                [np.nan, np.nan, -2.15, np.nan],
            ],
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, -2.45, np.nan],
                [np.nan, np.nan, -2.0, np.nan],
                [np.nan, np.nan, np.nan, -2.95],
            ],
            [
                [np.nan, np.nan, np.nan, -3.15],
                [np.nan, np.nan, -1.95, -3.15],
                [np.nan, np.nan, -2.0, np.nan],
                [np.nan, np.nan, np.nan, -2.95],
            ],
        ],
    )
    # Check that the other variables are also sliced correctly
    assert_array_equal(sliced["top"].notnull(), sliced["thickness"].notnull())
    assert_array_equal(sliced["top"].notnull(), sliced["kh"].notnull())

    # Test with only upper bound
    sliced = layermodels.slice_depth_interval(layermodel, upper=depth_grid)
    assert_array_equal(sliced["layer"], ["B", "C", "D"])
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [-0.5, np.nan, -1.05],
                [-0.5, np.nan, -0.85],
                [-0.8, -0.95, -2.55],
                [np.nan, -0.7, -2.15],
            ],
            [
                [-0.8, np.nan, -1.05],
                [-0.8, np.nan, -0.85],
                [-0.8, -0.95, -2.55],
                [np.nan, -0.8, -2.15],
            ],
            [
                [-0.7, np.nan, -1.05],
                [-0.7, -0.85, -2.45],
                [np.nan, -0.7, -2.0],
                [np.nan, np.nan, -0.7],
            ],
            [
                [np.nan, np.nan, -1.0],
                [np.nan, -1.0, -1.95],
                [np.nan, -0.6, -2.0],
                [np.nan, np.nan, -0.6],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [-1.05, np.nan, -3.25],
                [-0.85, np.nan, -3.25],
                [-0.95, -2.55, -3.35],
                [np.nan, -2.15, -3.35],
            ],
            [
                [-1.05, np.nan, -3.25],
                [-0.85, np.nan, -3.25],
                [-0.95, -2.55, -3.35],
                [np.nan, -2.15, -3.35],
            ],
            [
                [-1.05, np.nan, -3.25],
                [-0.85, -2.45, -3.25],
                [np.nan, -2.0, -3.2],
                [np.nan, np.nan, -2.95],
            ],
            [
                [np.nan, np.nan, -3.15],
                [np.nan, -1.95, -3.15],
                [np.nan, -2.0, -3.2],
                [np.nan, np.nan, -2.95],
            ],
        ],
    )
    # Test with only lower bound
    sliced = layermodels.slice_depth_interval(layermodel, lower=depth_grid)
    assert_array_equal(sliced["layer"], ["A", "B", "C", "D"])
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [0.2, -0.25, np.nan, np.nan],
                [0.3, -0.15, np.nan, np.nan],
                [0.25, -0.2, np.nan, np.nan],
                [0.1, np.nan, -0.35, np.nan],
            ],
            [
                [0.2, -0.25, np.nan, np.nan],
                [0.3, -0.15, np.nan, np.nan],
                [0.25, -0.2, np.nan, np.nan],
                [0.1, np.nan, -0.35, np.nan],
            ],
            [
                [0.2, -0.25, np.nan, np.nan],
                [0.3, -0.15, np.nan, np.nan],
                [0.25, np.nan, -0.2, np.nan],
                [0.1, np.nan, np.nan, -0.35],
            ],
            [
                [0.2, np.nan, np.nan, -0.25],
                [0.3, np.nan, -0.15, np.nan],
                [0.25, np.nan, -0.2, np.nan],
                [0.1, np.nan, np.nan, -0.35],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [-0.25, -0.5, np.nan, np.nan],
                [-0.15, -0.5, np.nan, np.nan],
                [-0.2, -0.8, np.nan, np.nan],
                [-0.35, np.nan, -0.7, np.nan],
            ],
            [
                [-0.25, -0.8, np.nan, np.nan],
                [-0.15, -0.8, np.nan, np.nan],
                [-0.2, -0.8, np.nan, np.nan],
                [-0.35, np.nan, -0.8, np.nan],
            ],
            [
                [-0.25, -0.7, np.nan, np.nan],
                [-0.15, -0.7, np.nan, np.nan],
                [-0.2, np.nan, -0.7, np.nan],
                [-0.35, np.nan, np.nan, -0.7],
            ],
            [
                [-0.25, np.nan, np.nan, -1.0],
                [-0.15, np.nan, -1.0, np.nan],
                [-0.2, np.nan, -0.6, np.nan],
                [-0.35, np.nan, np.nan, -0.6],
            ],
        ],
    )


@pytest.mark.unittest
def test_slice_depth_interval_with_1d_dataarray(layermodel):
    da_1d = xr.DataArray(
        np.full((4,), -0.4), coords={"x": [0.5, 1.5, 2.5, 3.5]}, dims=["x"]
    )
    sliced = layermodel.gst.slice_depth_interval(upper=da_1d, lower=da_1d - 1.2)
    assert isinstance(sliced, xr.Dataset)
    assert sliced.sizes == {"y": 4, "x": 4, "layer": 3}
    assert_array_equal(sliced.data_vars, layermodel.data_vars)
    assert_array_equal(sliced["layer"], ["B", "C", "D"])
    assert_array_equal(sliced["surface"], layermodel["surface"])  # Should be unchanged
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [-0.4, np.nan, -1.05],
                [-0.4, np.nan, -0.85],
                [-0.4, -0.95, np.nan],
                [np.nan, -0.4, np.nan],
            ],
            [
                [-0.4, np.nan, -1.05],
                [-0.4, np.nan, -0.85],
                [-0.4, -0.95, np.nan],
                [np.nan, -0.4, np.nan],
            ],
            [
                [-0.4, np.nan, -1.05],
                [-0.4, -0.85, np.nan],
                [np.nan, -0.4, np.nan],
                [np.nan, np.nan, -0.4],
            ],
            [
                [np.nan, np.nan, -0.4],
                [np.nan, -0.4, np.nan],
                [np.nan, -0.4, np.nan],
                [np.nan, np.nan, -0.4],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [-1.05, np.nan, -1.6],
                [-0.85, np.nan, -1.6],
                [-0.95, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
            ],
            [
                [-1.05, np.nan, -1.6],
                [-0.85, np.nan, -1.6],
                [-0.95, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
            ],
            [
                [-1.05, np.nan, -1.6],
                [-0.85, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
                [np.nan, np.nan, -1.6],
            ],
            [
                [np.nan, np.nan, -1.6],
                [np.nan, -1.6, np.nan],
                [np.nan, -1.6, np.nan],
                [np.nan, np.nan, -1.6],
            ],
        ],
    )
    # Check that the other variables are also sliced correctly
    assert_array_equal(sliced["top"].notnull(), sliced["thickness"].notnull())
    assert_array_equal(sliced["top"].notnull(), sliced["kh"].notnull())

    # Test update_top_bottom=False and drop=False --> keeps original shape and values
    sliced = layermodel.gst.slice_depth_interval(
        upper=da_1d, lower=da_1d - 1.2, update_top_bottom=False, drop=False
    )
    assert isinstance(sliced, xr.Dataset)
    assert sliced.sizes == {"y": 4, "x": 4, "layer": 4}
    assert_array_equal(sliced["layer"], layermodel["layer"])
    assert_array_equal(sliced["surface"], layermodel["surface"])  # Should be unchanged
    assert_array_almost_equal(
        sliced["top"],
        [
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, np.nan, -0.85],
                [np.nan, -0.2, -0.95, np.nan],
                [np.nan, np.nan, -0.35, np.nan],
            ],
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, np.nan, -0.85],
                [np.nan, -0.2, -0.95, np.nan],
                [np.nan, np.nan, -0.35, np.nan],
            ],
            [
                [np.nan, -0.25, np.nan, -1.05],
                [np.nan, -0.15, -0.85, np.nan],
                [np.nan, np.nan, -0.2, np.nan],
                [np.nan, np.nan, np.nan, -0.35],
            ],
            [
                [np.nan, np.nan, np.nan, -0.25],
                [np.nan, np.nan, -0.15, np.nan],
                [np.nan, np.nan, -0.2, np.nan],
                [np.nan, np.nan, np.nan, -0.35],
            ],
        ],
    )
    assert_array_almost_equal(
        sliced["bottom"],
        [
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, np.nan, -3.25],
                [np.nan, -0.95, -2.55, np.nan],
                [np.nan, np.nan, -2.15, np.nan],
            ],
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, np.nan, -3.25],
                [np.nan, -0.95, -2.55, np.nan],
                [np.nan, np.nan, -2.15, np.nan],
            ],
            [
                [np.nan, -1.05, np.nan, -3.25],
                [np.nan, -0.85, -2.45, np.nan],
                [np.nan, np.nan, -2.0, np.nan],
                [np.nan, np.nan, np.nan, -2.95],
            ],
            [
                [np.nan, np.nan, np.nan, -3.15],
                [np.nan, np.nan, -1.95, np.nan],
                [np.nan, np.nan, -2.0, np.nan],
                [np.nan, np.nan, np.nan, -2.95],
            ],
        ],
    )
    # Check that the other variables are also sliced correctly
    assert_array_equal(sliced["top"].notnull(), sliced["thickness"].notnull())
    assert_array_equal(sliced["top"].notnull(), sliced["kh"].notnull())
