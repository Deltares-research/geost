import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from geost.models._core import ModelType
from geost.models.model_dataset import ModelDataset


@pytest.fixture
def invalid_model(voxelmodel):
    """
    A model is invalid if it has both a voxelmodel ("z") and layermodel ("layer") vertical
    dimension.

    """
    invalid = voxelmodel.copy()
    invalid = invalid.assign_coords(layer=list(range(invalid.sizes["z"])))
    return invalid


class TestModelDataset:
    """
    Testing of accessor functionality on Datasets. Tests should exclusively use Dataset
    fixtures.
    """

    @pytest.mark.unittest
    def test_accessor_voxelmodel(self, voxelmodel):
        assert hasattr(voxelmodel, "gst")
        assert isinstance(voxelmodel.gst, ModelDataset)
        assert voxelmodel.gst._x == "x"
        assert voxelmodel.gst._y == "y"
        assert voxelmodel.gst._z == "z"
        assert voxelmodel.gst._model_type == ModelType.VOXEL
        assert voxelmodel.gst._top is None
        assert voxelmodel.gst._bottom is None
        assert voxelmodel.gst._zmin is None
        assert voxelmodel.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_layermodel(self, layermodel):
        assert hasattr(layermodel, "gst")
        assert isinstance(layermodel.gst, ModelDataset)
        assert layermodel.gst._x == "x"
        assert layermodel.gst._y == "y"
        assert layermodel.gst._z == "layer"
        assert layermodel.gst._model_type == ModelType.LAYER
        assert layermodel.gst._top == "top"
        assert layermodel.gst._bottom == "bottom"
        assert layermodel.gst._zmin is None
        assert layermodel.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_empty_dataset(self):
        ds = xr.Dataset()
        assert hasattr(ds, "gst")
        assert isinstance(ds.gst, ModelDataset)
        assert ds.gst._x is None
        assert ds.gst._y is None
        assert ds.gst._z is None
        assert ds.gst._model_type is None
        assert ds.gst._top is None
        assert ds.gst._bottom is None
        assert ds.gst._zmin is None
        assert ds.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_invalid_model(self, invalid_model):
        with pytest.raises(
            ValueError,
            match="Ambiguous vertical dimension: voxel=z, layer=layer",
        ):
            invalid_model.gst

    @pytest.mark.unittest
    def test_ndims(self, voxelmodel, layermodel):
        assert voxelmodel.gst.ndims == 3
        assert layermodel.gst.ndims == 3

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
        lower, upper = voxelmodel.gst.vertical_bounds()
        assert lower == pytest.approx(-2.5)
        assert upper == pytest.approx(0.0)

        lower, upper = layermodel.gst.vertical_bounds()
        assert lower == pytest.approx(-3.35)
        assert upper == pytest.approx(0.3)

    @pytest.mark.unittest
    def test_select_within_bbox(self, voxelmodel):
        sel = voxelmodel.gst.select_within_bbox(1, 1, 3, 3)
        assert isinstance(sel, xr.Dataset)
        assert sel.gst.shape == (2, 2, 5)
        assert_array_equal(sel["x"].values, [1.5, 2.5])
        assert_array_equal(sel["y"].values, [2.5, 1.5])

        with pytest.raises(
            ValueError, match="No data found within the specified bounding box"
        ):
            voxelmodel.gst.select_within_bbox(1, 1, 3, 3, crs=4326)

    @pytest.mark.unittest
    def test_mask_geometries_points(self, voxelmodel, layermodel, points):
        """
        All behaviour for crs, all_touched, invert and drop keyword arguments is tested
        in `TestModelDataArray.test_mask_geometries_points`.

        """
        masked = voxelmodel.gst.mask_geometries(points)
        assert isinstance(masked, xr.Dataset)
        assert masked.gst.shape == voxelmodel.gst.shape

        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(2.5, 2.5), (0.5, 0.5), (0.5, 1.5)], names=["y", "x"]
        )
        # Check if all variables are correctly masked
        removed_xy_cells = masked.isnull().all(dim="z")
        for var_ in masked.data_vars:
            removed = removed_xy_cells[var_]
            assert_array_equal(
                removed,
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, True, True, True],
                    [False, False, True, True],
                ],
            )
            not_missing = (
                ~removed
            ).to_series()  # Created Series contains the coordinates of the cells as the index
            assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        # Also test for layermodel, the same cells should be masked
        masked = layermodel.gst.mask_geometries(points)
        assert isinstance(masked, xr.Dataset)
        assert masked.gst.shape == layermodel.gst.shape

        removed_xy_cells = masked.isnull().all(dim="layer")
        for var_ in masked.data_vars:
            removed = removed_xy_cells[var_]
            assert_array_equal(
                removed,
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, True, True, True],
                    [False, False, True, True],
                ],
            )
            not_missing = (
                ~removed
            ).to_series()  # Created Series contains the coordinates of the cells as the index
            assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

    @pytest.mark.unittest
    def test_mask_geometries_lines(self, voxelmodel, layermodel, lines):
        """
        All behaviour for crs, all_touched, invert and drop keyword arguments is tested
        in `TestModelDataArray.test_mask_geometries_lines`.

        """
        masked = voxelmodel.gst.mask_geometries(lines)
        assert isinstance(masked, xr.Dataset)
        assert masked.gst.shape == voxelmodel.gst.shape
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(2.5, 2.5), (1.5, 1.5), (1.5, 3.5), (0.5, 0.5), (0.5, 2.5)],
            names=["y", "x"],
        )
        # Check if all variables are correctly masked
        removed_xy_cells = masked.isnull().all(dim="z")
        for var_ in masked.data_vars:
            removed = removed_xy_cells[var_]
            assert_array_equal(
                removed,
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, False, True, False],
                    [False, True, False, True],
                ],
            )
            not_missing = (
                ~removed
            ).to_series()  # Created Series contains the coordinates of the cells as the index
            assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        # Also test for layermodel, the same cells should be masked
        masked = layermodel.gst.mask_geometries(lines)
        assert isinstance(masked, xr.Dataset)
        assert masked.gst.shape == layermodel.gst.shape
        removed_xy_cells = masked.isnull().all(dim="layer")
        for var_ in masked.data_vars:
            removed = removed_xy_cells[var_]
            assert_array_equal(
                removed,
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, False, True, False],
                    [False, True, False, True],
                ],
            )
            not_missing = (
                ~removed
            ).to_series()  # Created Series contains the coordinates of the cells as the index
            assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

    @pytest.mark.unittest
    def test_mask_geometries_polygons(self, voxelmodel, layermodel, polygons):
        """
        All behaviour for crs, all_touched, invert and drop keyword arguments is tested
        in `TestModelDataArray.test_mask_geometries_polygons`.

        """
        masked = voxelmodel.gst.mask_geometries(polygons)
        assert isinstance(masked, xr.Dataset)
        assert masked.gst.shape == voxelmodel.gst.shape
        expected_xy_cells = pd.MultiIndex.from_tuples(
            [(1.5, 0.5), (1.5, 1.5), (0.5, 0.5), (0.5, 1.5)], names=["y", "x"]
        )
        # Check if all variables are correctly masked
        removed_xy_cells = masked.isnull().all(dim="z")
        for var_ in masked.data_vars:
            removed = removed_xy_cells[var_]
            assert_array_equal(
                removed,
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, True, True],
                    [False, False, True, True],
                ],
            )
            not_missing = (
                ~removed
            ).to_series()  # Created Series contains the coordinates of the cells as the index
            assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

        # Also test for layermodel, the same cells should be masked
        masked = layermodel.gst.mask_geometries(polygons)
        assert isinstance(masked, xr.Dataset)
        assert masked.gst.shape == layermodel.gst.shape
        removed_xy_cells = masked.isnull().all(dim="layer")
        for var_ in masked.data_vars:
            removed = removed_xy_cells[var_]
            assert_array_equal(
                removed,
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, False, True, True],
                    [False, False, True, True],
                ],
            )
            not_missing = (
                ~removed
            ).to_series()  # Created Series contains the coordinates of the cells as the index
            assert_array_equal(not_missing[not_missing].index, expected_xy_cells)

    @pytest.mark.unittest
    def test_select_points(self, voxelmodel, layermodel, points):
        """
        All behaviour for crs and drop keyword arguments is tested in
        `TestModelDataArray.test_select_points`.

        """
        selected = voxelmodel.gst.select_points(points)
        assert isinstance(selected, xr.Dataset)
        assert selected.sizes == {"idx": 3, "z": 5}
        assert_array_equal(selected["x"].values, [0.5, 2.5, 1.5])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5])

        # Also test for layermodel, the same points should be selected
        selected = layermodel.gst.select_points(points)
        assert isinstance(selected, xr.Dataset)
        assert selected.sizes == {"idx": 3, "layer": 4}
        assert_array_equal(selected["x"].values, [0.5, 2.5, 1.5])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5])

    @pytest.mark.unittest
    def test_select_along_lines(self, voxelmodel, layermodel, lines):
        """
        All behaviour for crs, distance and drop keyword arguments is tested in
        `TestModelDataArray.test_select_along_lines`.

        """
        expected_x_coords = [[0.5, 1.5, 2.5], [np.nan, 2.5, 3.5], [3.5, np.nan, np.nan]]
        expected_y_coords = [[0.5, 1.5, 2.5], [np.nan, 0.5, 0.5], [1.5, np.nan, np.nan]]
        selected = voxelmodel.gst.select_along_lines(lines)
        assert isinstance(selected, xr.Dataset)
        assert selected.sizes == {"line": 3, "distance": 3, "z": 5}
        assert_array_equal(selected["x"], expected_x_coords)
        assert_array_equal(selected["y"], expected_y_coords)

        # Also test for layermodel, the same points should be selected
        selected = layermodel.gst.select_along_lines(lines)
        assert isinstance(selected, xr.Dataset)
        assert selected.sizes == {"line": 3, "distance": 3, "layer": 4}
        assert_array_equal(selected["x"], expected_x_coords)
        assert_array_equal(selected["y"], expected_y_coords)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, voxelmodel, layermodel):
        pass
