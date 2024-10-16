import numpy as np
import pytest
import rioxarray
import xarray as xr

from geost.models.basemodels import VoxelModel


@pytest.fixture
def xarray_dataset():
    x = np.arange(4) + 0.5
    y = x[::-1]
    z = np.arange(0, 2, 0.5) + 0.25

    strat = [
        [[2, 2, 2, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 2, 1, 1]],
        [[2, 2, 1, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1]],
    ]
    lith = [
        [[2, 3, 2, 1], [2, 3, 1, 1], [2, 1, 1, 1], [3, 2, 1, 1]],
        [[2, 2, 1, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 3, 2, 1]],
        [[2, 3, 2, 1], [2, 1, 1, 3], [2, 2, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 3], [3, 2, 2, 1]],
    ]
    ds = xr.Dataset(
        data_vars=dict(strat=(["y", "x", "z"], strat), lith=(["y", "x", "z"], lith)),
        coords=dict(y=y, x=x, z=z),
    )
    ds.rio.write_crs(28992, inplace=True)
    return ds


@pytest.fixture
def voxelmodel(xarray_dataset):
    return VoxelModel(xarray_dataset)
