import numpy as np
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


@pytest.fixture
def layermodel_top(layermodel):
    return layermodel["top"]


@pytest.fixture
def layermodel_bottom(layermodel):
    return layermodel["bottom"]


@pytest.fixture
def invalid_dataarray():
    return xr.DataArray(
        np.zeros((2, 3, 4, 5)),
        dims=("y", "x", "z", "layer"),
    )


class TestModelDataArray:
    """
    Testing of accessor functionality on DataArrays. Tests should exclusively use
    DataArray fixtures.
    """

    @pytest.mark.unittest
    def test_accessor_voxelmodel(self, voxelmodel_var):
        assert hasattr(voxelmodel_var, "gst")
        assert isinstance(voxelmodel_var.gst, ModelDataArray)
        assert voxelmodel_var.gst._x == "x"
        assert voxelmodel_var.gst._y == "y"
        assert voxelmodel_var.gst._z == "z"
        assert voxelmodel_var.gst._model_type == ModelType.VOXEL
        assert voxelmodel_var.gst._top is None
        assert voxelmodel_var.gst._bottom is None
        assert voxelmodel_var.gst._zmin is None
        assert voxelmodel_var.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_layermodel(self, layermodel_var):
        assert hasattr(layermodel_var, "gst")
        assert isinstance(layermodel_var.gst, ModelDataArray)
        assert layermodel_var.gst._x == "x"
        assert layermodel_var.gst._y == "y"
        assert layermodel_var.gst._z == "layer"
        assert layermodel_var.gst._model_type == ModelType.LAYER

        # A variable in a layermodel does not have top and bottom because these are Dataset-level
        # attributes. _top or _bottom will only be set if the DataArray is the top or bottom
        # variable of the layermodel itself, see unittest test_accessor_layermodel_top.
        assert layermodel_var.gst._top is None
        assert layermodel_var.gst._bottom is None
        assert layermodel_var.gst._zmin is None
        assert layermodel_var.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_layermodel_top(self, layermodel_top):
        assert hasattr(layermodel_top, "gst")
        assert isinstance(layermodel_top.gst, ModelDataArray)
        assert layermodel_top.gst._x == "x"
        assert layermodel_top.gst._y == "y"
        assert layermodel_top.gst._z == "layer"
        assert layermodel_top.gst._model_type == ModelType.LAYER
        assert layermodel_top.gst._top == "top"
        assert layermodel_top.gst._bottom is None
        assert layermodel_top.gst._zmin is None
        assert layermodel_top.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_layermodel_bottom(self, layermodel_bottom):
        assert layermodel_bottom.gst._top is None
        assert layermodel_bottom.gst._bottom == "bottom"

    @pytest.mark.unittest
    def test_accessor_empty_dataarray(self):
        da = xr.DataArray()
        assert hasattr(da, "gst")
        assert isinstance(da.gst, ModelDataArray)
        assert da.gst._x is None
        assert da.gst._y is None
        assert da.gst._z is None
        assert da.gst._model_type is None
        assert da.gst._top is None
        assert da.gst._bottom is None
        assert da.gst._zmin is None
        assert da.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_invalid_array(self, invalid_dataarray):
        with pytest.raises(
            ValueError,
            match="Ambiguous vertical dimension: voxel=z, layer=layer",
        ):
            invalid_dataarray.gst

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
        with pytest.raises(
            TypeError,
            match="Vertical bounds cannot be determined for a layermodel DataArray. This method "
            "can only be used on an xarray.Dataset with valid 'top' and 'bottom' variables.",
        ):
            layermodel_var.gst.vertical_bounds()
