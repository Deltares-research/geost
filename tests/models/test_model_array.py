import pytest
import xarray as xr

from geost.models._core import ModelType
from geost.models.model_array import ModelDataArray


@pytest.fixture
def voxelmodel_var(voxelmodel):
    return voxelmodel["strat"]


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

    @pytest.mark.unittest
    def test_crs(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.crs == 28992
        assert layermodel_var.gst.crs == 28992

    @pytest.mark.unittest
    def test_x_dim(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.x_dim == "x"
        assert layermodel_var.gst.x_dim == "x"

    @pytest.mark.unittest
    def test_y_dim(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.y_dim == "y"
        assert layermodel_var.gst.y_dim == "y"

    @pytest.mark.unittest
    def test_z_dim(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.z_dim == "z"
        assert layermodel_var.gst.z_dim == "layer"

    @pytest.mark.unittest
    def test_model_type(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.model_type == ModelType.VOXEL
        assert layermodel_var.gst.model_type == ModelType.LAYER

    @pytest.mark.unittest
    def test_shape(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.shape == (4, 4, 5)
        assert layermodel_var.gst.shape == (4, 4, 4)

    @pytest.mark.unittest
    def test_resolution(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.resolution() == (1.0, -1.0, 0.5)
        assert layermodel_var.gst.resolution() == (1.0, -1.0)

        with pytest.raises(
            ValueError, match="Resolution cannot be determined for 1D models."
        ):
            layermodel_var.isel(x=0, y=0).gst.resolution()

        with pytest.raises(
            ValueError, match="Resolution cannot be determined for 1D models."
        ):
            voxelmodel_var.isel(x=[0], y=[0]).gst.resolution()

    @pytest.mark.unittest
    def test_vertical_bounds(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.vertical_bounds() == (-2.5, 0.0)
        assert layermodel_var.gst.vertical_bounds() == (-3.35, 0.3)
