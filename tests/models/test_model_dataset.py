import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.exceptions import InvalidModelError, ModelTypeError
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
        assert_array_equal(voxelmodel.gst.x, voxelmodel["x"])
        assert_array_equal(voxelmodel.gst.y, voxelmodel["y"])
        assert_array_equal(voxelmodel.gst.z, voxelmodel["z"])

        with pytest.raises(
            ModelTypeError, match="Only ModelType.LAYER has a 'top' property."
        ):
            voxelmodel.gst.top
        with pytest.raises(
            ModelTypeError, match="Only ModelType.LAYER has a 'bottom' property."
        ):
            voxelmodel.gst.bottom

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
        assert_array_equal(layermodel.gst.x, layermodel["x"])
        assert_array_equal(layermodel.gst.y, layermodel["y"])
        assert_array_equal(layermodel.gst.z, layermodel["layer"])
        assert_array_equal(layermodel.gst.top, layermodel["top"])
        assert_array_equal(layermodel.gst.bottom, layermodel["bottom"])

    @pytest.mark.unittest
    def test_accessor_empty_dataset(self):
        ds = xr.Dataset()
        error = (
            "Invalid model: \n"
            "Missing x and/or y dimensions.\n"
            "Missing z dimension for voxelmodel or top/bottom for layermodel."
        )
        with pytest.raises(InvalidModelError, match=error):
            ds.gst

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

    @pytest.mark.parametrize(
        "depth",
        [
            -0.4,
            np.full((4, 4), -0.4),
            xr.DataArray(
                np.full((4, 4), -0.4),
                coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]},
                dims=["y", "x"],
            ),
            xr.DataArray(
                np.full((4,), -0.4), coords={"x": [0.5, 1.5, 2.5, 3.5]}, dims=["x"]
            ),
        ],
        ids=["scalar", "2D-array", "2D-dataarray", "1D-dataarray"],
    )
    def test_slice_depth_interval(self, voxelmodel, layermodel, depth):
        """
        All behaviour for upper, lower, how, update_top_bottom and drop keyword arguments is
        tested in the module tests/models/test_voxelmodels.py and tests/models/test_layermodels.py.

        """
        sliced = voxelmodel.gst.slice_depth_interval(upper=depth, lower=depth - 1.2)
        assert isinstance(sliced, xr.Dataset)
        assert sliced.sizes == {"y": 4, "x": 4, "z": 4}
        assert_array_equal(sliced.data_vars, voxelmodel.data_vars)
        assert_array_equal(sliced["z"], [-1.75, -1.25, -0.75, -0.25])

        sliced = layermodel.gst.slice_depth_interval(upper=depth, lower=depth - 1.2)
        assert isinstance(sliced, xr.Dataset)
        assert sliced.sizes == {"y": 4, "x": 4, "layer": 3}
        assert_array_equal(sliced["layer"], ["B", "C", "D"])
        assert_array_equal(sliced.data_vars, layermodel.data_vars)
        assert_array_equal(sliced["surface"], layermodel["surface"])

    @pytest.mark.unittest
    def test_most_common_voxelmodel(self, voxelmodel):
        expected_mode_strat = [
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0, 2.0],
        ]
        expected_mode_lith = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
        ]
        expected_thickness_strat = [
            [1.5, 1.0, 1.5, 1.0],
            [1.5, 1.5, 1.5, 1.0],
            [1.5, 2.0, 1.5, 1.0],
            [1.5, 2.0, 1.0, 1.5],
        ]
        expected_thickness_lith = [
            [1.0, 1.0, 1.5, 1.0],
            [1.5, 1.5, 1.5, 1.0],
            [1.0, 1.0, 1.5, 1.0],
            [1.5, 2.0, 1.0, 1.0],
        ]
        result = voxelmodel.gst.most_common()
        assert isinstance(result, xr.DataArray)
        assert result.sizes == {"data_var": 2, "y": 4, "x": 4}
        assert_array_equal(result.sel(data_var="strat"), expected_mode_strat)
        assert_array_equal(result.sel(data_var="lith"), expected_mode_lith)

        result = voxelmodel.gst.most_common(return_thickness=True)
        assert isinstance(result, xr.Dataset)
        assert result.sizes == {"data_var": 2, "y": 4, "x": 4}
        assert_array_equal(
            result["most_common"].sel(data_var="strat"), expected_mode_strat
        )
        assert_array_equal(
            result["most_common"].sel(data_var="lith"), expected_mode_lith
        )
        assert_array_equal(
            result["thickness"].sel(data_var="strat"), expected_thickness_strat
        )
        assert_array_equal(
            result["thickness"].sel(data_var="lith"), expected_thickness_lith
        )

    @pytest.mark.unittest
    def test_most_common_layermodel(self, layermodel):
        expected_mode = [
            [85.0, 85.0, 20.1, 20.1],
            [85.0, 85.0, 20.1, 20.1],
            [85.0, 20.1, 20.1, 85.0],
            [85.0, 20.1, 20.1, 85.0],
        ]
        expected_most_common_layer = [
            ["D", "D", "C", "C"],
            ["D", "D", "C", "C"],
            ["D", "C", "C", "D"],
            ["D", "C", "C", "D"],
        ]
        expected_thickness = [
            [2.2, 2.4, 1.6, 1.8],
            [2.2, 2.4, 1.6, 1.8],
            [2.2, 1.6, 1.8, 2.6],
            [2.9, 1.8, 1.8, 2.6],
        ]
        expected_top = [
            [-1.05, -0.85, -0.95, -0.35],
            [-1.05, -0.85, -0.95, -0.35],
            [-1.05, -0.85, -0.2, -0.35],
            [-0.25, -0.15, -0.2, -0.35],
        ]
        expected_bottom = [
            [-3.25, -3.25, -2.55, -2.15],
            [-3.25, -3.25, -2.55, -2.15],
            [-3.25, -2.45, -2.0, -2.95],
            [-3.15, -1.95, -2.0, -2.95],
        ]
        result = layermodel.gst.most_common()
        assert isinstance(result, xr.Dataset)
        assert result.sizes == {"x": 4, "y": 4}
        assert_array_equal(
            result.data_vars,
            [
                "most_common_layer",
                "top_most_common",
                "bottom_most_common",
                "thickness_most_common",
                "kh_most_common",
            ],
        )
        assert_array_equal(result["most_common_layer"], expected_most_common_layer)
        assert_array_almost_equal(result["top_most_common"], expected_top)
        assert_array_almost_equal(result["bottom_most_common"], expected_bottom)
        assert_array_almost_equal(result["thickness_most_common"], expected_thickness)
        assert_array_almost_equal(result["kh_most_common"], expected_mode)

        result = layermodel.gst.most_common(return_thickness=True)
        assert isinstance(result, xr.Dataset)
        assert result.sizes == {"x": 4, "y": 4}
        assert_array_equal(
            result.data_vars,
            [
                "most_common_layer",
                "top_most_common",
                "bottom_most_common",
                "thickness_most_common",
                "kh_most_common",
                "thickness",
            ],
        )
        assert_array_equal(result["most_common_layer"], expected_most_common_layer)
        assert_array_almost_equal(result["top_most_common"], expected_top)
        assert_array_almost_equal(result["bottom_most_common"], expected_bottom)
        assert_array_almost_equal(result["thickness_most_common"], expected_thickness)
        assert_array_almost_equal(result["kh_most_common"], expected_mode)
        assert_array_almost_equal(result["thickness"], expected_thickness)
        assert_array_almost_equal(result["thickness"], result["thickness_most_common"])
