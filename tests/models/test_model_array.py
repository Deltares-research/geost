import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

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


@pytest.fixture
def dataarray_inverted_dims():
    x = np.arange(0, 4) + 0.5
    y = np.arange(3, 0, -1) - 0.5
    z = [0.75, 0.25]
    da = xr.DataArray(
        np.ones((4, 3, 2)), dims=("x", "y", "z"), coords={"x": x, "y": y, "z": z}
    )
    da.rio.write_crs("EPSG:28992", inplace=True)
    return da


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
    def test_ndims(self, voxelmodel_var, layermodel_var):
        assert voxelmodel_var.gst.ndims == 3
        assert layermodel_var.gst.ndims == 3

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

    @pytest.mark.unittest
    def test_select_within_bbox(self, voxelmodel_var):
        sel = voxelmodel_var.gst.select_within_bbox(1, 1, 3, 3)
        assert isinstance(sel, xr.DataArray)
        assert sel.gst.shape == (2, 2, 5)
        assert_array_equal(sel["x"].values, [1.5, 2.5])
        assert_array_equal(sel["y"].values, [2.5, 1.5])

        with pytest.raises(
            ValueError, match="No data found within the specified bounding box"
        ):
            voxelmodel_var.gst.select_within_bbox(1, 1, 3, 3, crs=4326)

    @pytest.mark.unittest
    def test_mask_geometries_points(self, voxelmodel_var, points):
        masked = voxelmodel_var.gst.mask_geometries(points)
        assert isinstance(masked, xr.DataArray)
        assert masked.gst.shape == voxelmodel_var.gst.shape
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
                [False, False, True, True],
            ],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(2.5, 2.5), (0.5, 0.5), (0.5, 1.5)], names=["y", "x"]
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        masked = voxelmodel_var.gst.mask_geometries(points, all_touched=True)
        not_missing = (~removed_xy_cells).to_series()
        assert_array_equal(
            not_missing[not_missing].index, expected_xy_cells
        )  # all_touched=True does not change the result for points, so same expected_xy_cells

        # Test invert=True, which should result in the inverse of the previous mask
        masked = voxelmodel_var.gst.mask_geometries(points, invert=True)
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [False, False, False, False],
                [False, False, True, False],
                [False, False, False, False],
                [True, True, False, False],
            ],
        )

        masked = voxelmodel_var.gst.mask_geometries(points, crs=4326)
        # CRS of the voxelmodel differs from the CRS of the points, so nothing is burned in
        # in the resulting masked DataArray.
        assert masked.isnull().all()

        masked = voxelmodel_var.gst.mask_geometries(points, crs=4326, invert=True)
        # Invert=True should result in the inverse of the previous mask, so the result
        # should be the same as the original voxelmodel_var.
        assert masked.equals(voxelmodel_var)

        # Test drop=True, which should remove full rows and columns that are completely masked out
        masked = voxelmodel_var.gst.mask_geometries(points, drop=True)
        assert masked.sizes == {"y": 3, "x": 3, "z": 5}
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [[True, True, False], [True, True, True], [False, False, True]],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(2.5, 2.5), (0.5, 0.5), (0.5, 1.5)], names=["y", "x"]
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

    @pytest.mark.unittest
    def test_mask_geometries_lines(self, voxelmodel_var, lines):
        masked = voxelmodel_var.gst.mask_geometries(lines)
        assert isinstance(masked, xr.DataArray)
        assert masked.gst.shape == voxelmodel_var.gst.shape
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
            ],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(2.5, 2.5), (1.5, 1.5), (1.5, 3.5), (0.5, 0.5), (0.5, 2.5)],
            names=["y", "x"],
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        # Test all_touched=True and drop=True
        masked = voxelmodel_var.gst.mask_geometries(lines, all_touched=True, drop=True)
        assert masked.sizes == {"y": 3, "x": 4, "z": 5}
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [True, False, False, True],
                [False, False, True, False],
                [False, True, False, False],
            ],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [
                (2.5, 1.5),
                (2.5, 2.5),
                (1.5, 0.5),
                (1.5, 1.5),
                (1.5, 3.5),
                (0.5, 0.5),
                (0.5, 2.5),
                (0.5, 3.5),
            ],
            names=["y", "x"],
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        masked = voxelmodel_var.gst.mask_geometries(lines, crs=4326)
        # CRS of the voxelmodel differs from the CRS of the lines, so nothing is burned in
        # in the resulting masked DataArray.
        assert masked.isnull().all()

        masked = voxelmodel_var.gst.mask_geometries(lines, crs=4326, invert=True)
        # Invert=True should result in the inverse of the previous mask, so the result
        # should be the same as the original voxelmodel_var.
        assert masked.equals(voxelmodel_var)

        # Test invert=True
        masked = voxelmodel_var.gst.mask_geometries(lines, invert=True)
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [False, False, False, False],
                [False, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ],
        )

    def test_mask_geometries_polygons(self, voxelmodel_var, polygons):
        masked = voxelmodel_var.gst.mask_geometries(polygons)
        assert isinstance(masked, xr.DataArray)
        assert masked.gst.shape == voxelmodel_var.gst.shape
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(1.5, 0.5), (1.5, 1.5), (0.5, 0.5), (0.5, 1.5)], names=["y", "x"]
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        # Test drop=True: should remove rows, cols but select the same coordinates as the previous test
        masked = voxelmodel_var.gst.mask_geometries(polygons, drop=True)
        assert masked.sizes == {"y": 2, "x": 2, "z": 5}
        removed_xy_cells = masked.isnull().all(dim="z")
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        # Test invert=True
        masked = voxelmodel_var.gst.mask_geometries(polygons, invert=True)
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ],
        )

        # Test all_touched=True
        masked = voxelmodel_var.gst.mask_geometries(polygons, all_touched=True)
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [True, True, True, True],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, True],
            ],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [
                (2.5, 0.5),
                (2.5, 1.5),
                (2.5, 2.5),
                (2.5, 3.5),
                (1.5, 0.5),
                (1.5, 1.5),
                (1.5, 2.5),
                (1.5, 3.5),
                (0.5, 0.5),
                (0.5, 1.5),
                (0.5, 2.5),
            ],
            names=["y", "x"],
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        masked = voxelmodel_var.gst.mask_geometries(polygons, crs=4326)
        # CRS of the voxelmodel differs from the CRS of the polygons, so nothing is burned in
        # in the resulting masked DataArray.
        assert masked.isnull().all()

        masked = voxelmodel_var.gst.mask_geometries(polygons, crs=4326, invert=True)
        # Invert=True should result in the inverse of the previous mask, so the result
        # should be the same as the original voxelmodel_var.
        assert masked.equals(voxelmodel_var)

    @pytest.mark.unittest
    def test_mask_geometries_inverted_dims(self, dataarray_inverted_dims, points):
        # With inverted x,y dimensions, the mask result will be transposed. This is needed
        # for consistency in coordinates
        assert dataarray_inverted_dims.gst.shape == (4, 3, 2)
        masked = dataarray_inverted_dims.gst.mask_geometries(points)
        assert masked.gst.shape == (3, 4, 2)
        removed_xy_cells = masked.isnull().all(dim="z")
        assert_array_equal(
            removed_xy_cells,
            [
                [True, True, False, True],
                [True, True, True, True],
                [False, False, True, True],
            ],
        )
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(2.5, 2.5), (0.5, 0.5), (0.5, 1.5)], names=["y", "x"]
        )
        not_missing = (
            ~removed_xy_cells
        ).to_series()  # Created Series contains the coordinates of the cells as the index
        assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

    @pytest.mark.unittest
    def test_select_with_points(self, voxelmodel_var, layermodel_var, points):
        selected = voxelmodel_var.gst.select_points(points)
        assert isinstance(selected, xr.DataArray)
        assert selected.sizes == {"idx": 3, "z": 5}
        assert_array_equal(selected["idx"].values, [0, 1, 2])
        assert_array_equal(selected["x"].values, [0.5, 2.5, 1.5])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5])
        assert_array_equal(selected["z"].values, [-2.25, -1.75, -1.25, -0.75, -0.25])
        assert_array_equal(
            selected,
            [
                [2.0, 2.0, 2.0, 1.0, np.nan],
                [2.0, 1.0, 1.0, 1.0, np.nan],
                [2.0, 1.0, 1.0, 1.0, 1.0],
            ],
        )

        selected = layermodel_var.gst.select_points(points)
        assert isinstance(selected, xr.DataArray)
        assert selected.sizes == {"idx": 3, "layer": 4}
        assert_array_equal(selected["idx"].values, [0, 1, 2])
        assert_array_equal(selected["x"].values, [0.5, 2.5, 1.5])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5])
        assert_array_equal(selected["layer"].values, ["A", "B", "C", "D"])

        selected = voxelmodel_var.gst.select_points(points, drop=False)
        assert selected.sizes == {"idx": 4, "z": 5}
        assert_array_equal(selected["idx"].values, [0, 1, 2, 3])
        assert_array_equal(selected["x"].values, [0.5, 2.5, 1.5, np.nan])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5, np.nan])
        assert_array_equal(
            selected,
            [
                [2.0, 2.0, 2.0, 1.0, np.nan],
                [2.0, 1.0, 1.0, 1.0, np.nan],
                [2.0, 1.0, 1.0, 1.0, 1.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

        # Test that selecting points outside the model bounds results in an empty selection
        selected = voxelmodel_var.gst.select_points(points.to_crs(4326))
        assert selected.sizes == {"idx": 0, "z": 5}

        # Should produce the same result as the first test but differs due to coordinate
        # transformation (see below)
        selected = voxelmodel_var.gst.select_points(points.to_crs(4326), crs=4326)
        assert selected.sizes == {"idx": 3, "z": 5}
        assert_array_equal(selected["idx"].values, [0, 1, 2])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5])
        # The third point is exactly on the edge of a cell and now falls in the cell
        # with x=0.5 instead of x=1.5 due to rounding errors in coordinate transformation.
        # This is expected behavior.
        assert_array_equal(selected["x"].values, [0.5, 2.5, 0.5])
