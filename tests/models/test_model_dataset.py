import pytest
import xarray as xr

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

    @pytest.mark.unitest
    def test_z_dim(self, voxelmodel, layermodel):
        assert voxelmodel.gst.z_dim == "z"
        assert layermodel.gst.z_dim == "layer"
