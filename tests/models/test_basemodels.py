import pytest

from geost.models.basemodels import VoxelModel


class TestVoxelModel:
    @pytest.mark.unittest
    def test_initialize(self, xarray_dataset):
        model = VoxelModel(xarray_dataset)
        assert isinstance(model, VoxelModel)
        # TODO: Make sure an input DataArray becomes a Dataset which is needed by all methods.

    @pytest.mark.unittest
    def test_resolution(self, voxelmodel):
        resolution = voxelmodel.resolution
        assert resolution == (1, 1, 0.5)

    @pytest.mark.unittest
    def test_horizontal_bounds(self, voxelmodel):
        bounds = voxelmodel.horizontal_bounds
        assert bounds == (0, 0, 4, 4)

    @pytest.mark.unittest
    def test_vertical_bounds(self, voxelmodel):
        bounds = voxelmodel.vertical_bounds
        assert bounds == (0, 2)

    @pytest.mark.unittest
    def test_crs(self, voxelmodel):
        voxelmodel.crs == 28992