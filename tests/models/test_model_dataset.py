import pytest
import xarray as xr

from geost.models._core import ModelType
from geost.models.model_dataset import ModelDataset


class TestModelDataset:
    """
    Testing of accessor functionality on Datasets. Tests should exclusively use Dataset
    fixtures.
    """

    @pytest.mark.unittest
    def test_accessor(self):
        ds = xr.Dataset()
        assert hasattr(ds, "gst")
        assert isinstance(ds.gst, ModelDataset)

    @pytest.mark.unittest
    def test_crs(self, voxelmodel, layermodel):
        assert voxelmodel.gst.crs == 28992
        assert layermodel.gst.crs == 28992

    @pytest.mark.unittest
    def test_x_dim(self, voxelmodel, layermodel):
        assert voxelmodel.gst.x_dim == "x"
        assert layermodel.gst.x_dim == "x"

    @pytest.mark.unittest
    def test_y_dim(self, voxelmodel, layermodel):
        assert voxelmodel.gst.y_dim == "y"
        assert layermodel.gst.y_dim == "y"

    @pytest.mark.unittest
    def test_z_dim(self, voxelmodel, layermodel):
        assert voxelmodel.gst.z_dim == "z"
        assert layermodel.gst.z_dim == "layer"

        # For a layermodel, also a 'top' and 'bottom' should be detected
        assert layermodel.gst._top == "top"
        assert layermodel.gst._bottom == "bottom"

    @pytest.mark.unittest
    def test_model_type(self, voxelmodel, layermodel):
        assert voxelmodel.gst.model_type == ModelType.VOXEL
        assert layermodel.gst.model_type == ModelType.LAYER

    @pytest.mark.unittest
    def test_shape(self, voxelmodel, layermodel):
        assert voxelmodel.gst.shape == (4, 4, 5)
        assert layermodel.gst.shape == (4, 4, 4)

    @pytest.mark.unittest
    def test_resolution(self, voxelmodel, layermodel):
        assert voxelmodel.gst.resolution() == (1.0, -1.0, 0.5)
        assert layermodel.gst.resolution() == (1.0, -1.0)

        with pytest.raises(
            ValueError, match="Resolution cannot be determined for 1D models."
        ):
            layermodel.isel(x=[0], y=[0]).gst.resolution()

        with pytest.raises(
            ValueError, match="Resolution cannot be determined for 1D models."
        ):
            layermodel.isel(x=0, y=0).gst.resolution()

    @pytest.mark.unittest
    def test_vertical_bounds(self, voxelmodel, layermodel):
        assert voxelmodel.gst.vertical_bounds() == (-2.5, 0.0)
        assert layermodel.gst.vertical_bounds() == (-3.35, 0.3)
