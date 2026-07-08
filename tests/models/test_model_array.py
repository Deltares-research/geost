import pytest
import xarray as xr

from geost.models.model_array import ModelDataArray


@pytest.fixture
def voxelmodel_strat(voxelmodel):
    return voxelmodel["strat"]


@pytest.fixture
def voxelmodel_lith(voxelmodel):
    return voxelmodel["lith"]


@pytest.fixture
def layermodel_var(layermodel):
    return layermodel["kh"]


class TestModelDataArray:
    """
    Testing of accessor functionality on DataArrays. Tests should exclusively use
    DataArray fixtures.
    """

    @pytest.mark.unittest
    def test_accessor(self):
        da = xr.DataArray()
        assert hasattr(da, "gst")
        assert isinstance(da.gst, ModelDataArray)

    @pytest.mark.parametrize(
        "da, expected_z_dim",
        [
            ("voxelmodel_strat", "z"),
            ("voxelmodel_lith", "z"),
            ("layermodel_var", "layer"),
        ],
    )
    def test_z_dim(self, da, expected_z_dim, request):
        da = request.getfixturevalue(da)
        assert da.gst.z_dim == expected_z_dim
