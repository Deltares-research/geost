import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.exceptions import InvalidModelError
from geost.models._core import ModelType
from geost.models.model_array import ModelDataArray


@pytest.fixture
def voxelmodel_var(voxelmodel):
    return voxelmodel["strat"]


@pytest.fixture
def layermodel_var(layermodel):
    layermodel = layermodel.set_coords(
        ["top", "bottom"]
    )  # A DataArray with "top" and "bottom" coordinates is not a valid layermodel, so we need to set them as coordinates to get a valid layermodel DataArray
    return layermodel["kh"]


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


@pytest.fixture
def model_wgs(voxelmodel_var):
    new_x = [110_050, 110_150, 110_250, 110_350]
    new_y = [487_950, 487_850, 487_750, 487_650]
    coords = {"x": new_x, "y": new_y, "z": voxelmodel_var["z"].values}
    new = xr.DataArray(voxelmodel_var.data, coords=coords, dims=("y", "x", "z"))
    new.rio.write_crs(28992, inplace=True)
    return new.transpose("z", "y", "x").rio.reproject(4326)


@pytest.fixture
def lines_wgs():
    return gpd.GeoDataFrame(
        geometry=[
            shapely.LineString([(110_060, 487_970), (110_060, 487_870)]),
            shapely.LineString([(120_250, 487_950), (121_350, 487_650)]),
        ],
        crs=28992,
    ).to_crs(4326)


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
        assert layermodel_var.gst._top == "top"
        assert layermodel_var.gst._bottom == "bottom"
        assert layermodel_var.gst._zmin is None
        assert layermodel_var.gst._zmax is None

    @pytest.mark.unittest
    def test_accessor_empty_dataarray(self):
        da = xr.DataArray()
        error = (
            "Invalid model: \n"
            "Missing x and/or y dimensions.\n"
            "Missing z dimension for voxelmodel or top/bottom for layermodel."
        )
        with pytest.raises(InvalidModelError, match=error):
            da.gst

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

        model_wgs = voxelmodel_var.transpose("z", "y", "x").rio.reproject(4326)
        xres, yres, zres = model_wgs.gst.resolution()
        assert np.isclose(xres, 1.151273157390868e-05)
        assert np.isclose(yres, -1.151273157390868e-05)
        assert np.isclose(zres, 0.5)

        xres, yres, zres = model_wgs.gst.resolution(meters=True)
        assert np.isclose(xres, 0.9880535778432994)
        assert np.isclose(yres, -0.9880535778432994)
        assert np.isclose(zres, 0.5)

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

        # Also coordinates have no CRS but the same metric units can work
        selected = voxelmodel_var.gst.select_points(
            points.set_crs(None, allow_override=True)
        )
        assert selected.sizes == {"idx": 3, "z": 5}
        assert_array_equal(selected["idx"].values, [0, 1, 2])
        assert_array_equal(selected["x"].values, [0.5, 2.5, 1.5])
        assert_array_equal(selected["y"].values, [0.5, 2.5, 0.5])

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

    @pytest.mark.unittest
    def test_select_along_line(self, voxelmodel_var, layermodel_var, lines):
        # Define expected coordinates and values for the selection along the lines. These
        # must be the same for several selection results
        expected_x_coords = [
            [0.5, 1.5, 1.5, 1.5, 1.5, 2.5, np.nan],
            [np.nan, np.nan, np.nan, 2.5, 3.5, 3.5, 3.5],
            [3.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
        expected_y_coords = [
            [0.5, 1.5, 1.5, 1.5, 2.5, 2.5, np.nan],
            [np.nan, np.nan, np.nan, 0.5, 0.5, 0.5, 1.5],
            [1.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
        expected_selection_values = [
            [
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, np.nan],
                [np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, np.nan],
                [np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0],
                [2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, np.nan, np.nan, 1.0, 2.0, 2.0, 2.0],
                [2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
                [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 1.0, 1.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ]

        # Normal selection from voxelmodel
        selected = voxelmodel_var.gst.select_along_lines(lines, distance=0.4)
        assert isinstance(selected, xr.DataArray)
        assert selected.sizes == {"z": 5, "line": 3, "distance": 7}
        assert_array_almost_equal(
            selected["distance"], [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
        )
        assert_array_equal(
            selected["line"], [0, 1, 2]
        )  # line 3 is outside extent and is dropped
        assert_array_equal(selected["z"].values, [-2.25, -1.75, -1.25, -0.75, -0.25])
        assert selected["x"].dims == selected["y"].dims == ("line", "distance")
        assert_array_almost_equal(selected["x"], expected_x_coords)
        assert_array_almost_equal(selected["y"], expected_y_coords)
        assert_array_equal(selected, expected_selection_values)

        # Normal selection from layermodel
        selected = layermodel_var.gst.select_along_lines(lines, distance=0.4)
        assert selected.sizes == {"layer": 4, "line": 3, "distance": 7}
        assert_array_equal(selected["line"], [0, 1, 2])
        assert_array_almost_equal(
            selected["distance"], [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
        )
        assert_array_equal(selected["layer"].values, ["A", "B", "C", "D"])
        assert_array_almost_equal(selected["x"], expected_x_coords)
        assert_array_almost_equal(selected["y"], expected_y_coords)

        # Test selection with lines in other CRS
        selected = voxelmodel_var.gst.select_along_lines(
            lines.to_crs(4326), distance=0.4, crs=4326
        )
        assert selected.sizes == {"z": 5, "line": 3, "distance": 7}
        assert_array_almost_equal(
            selected["distance"], [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
        )
        assert_array_equal(selected["line"], [0, 1, 2])
        assert selected["x"].dims == selected["y"].dims == ("line", "distance")
        assert_array_almost_equal(selected["x"], expected_x_coords)
        assert_array_almost_equal(selected["y"], expected_y_coords)
        assert_array_equal(selected, expected_selection_values)

        # Also coordinates have no CRS but the same metric units can work
        selected = voxelmodel_var.gst.select_along_lines(
            lines.set_crs(None, allow_override=True), distance=0.4
        )
        assert selected.sizes == {"z": 5, "line": 3, "distance": 7}
        assert_array_almost_equal(
            selected["distance"], [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
        )
        assert_array_equal(selected["line"], [0, 1, 2])
        assert selected["x"].dims == selected["y"].dims == ("line", "distance")
        assert_array_almost_equal(selected["x"], expected_x_coords)
        assert_array_almost_equal(selected["y"], expected_y_coords)
        assert_array_equal(selected, expected_selection_values)

        # Outside the model extent, should return an empty selection
        selected = voxelmodel_var.gst.select_along_lines(
            lines.to_crs(4326), distance=0.4
        )
        assert selected.sizes == {"z": 5, "line": 0, "distance": 0}
        # Using drop is False should return with all line and distance coordinates but filled with NaN values
        selected = voxelmodel_var.gst.select_along_lines(
            lines.to_crs(4326), distance=0.4, drop=False
        )
        assert selected.sizes == {"z": 5, "line": 4, "distance": 8}
        assert_array_equal(selected["line"], [0, 1, 2, 3])
        assert_array_almost_equal(
            selected["distance"], [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
        )
        assert selected.isnull().all()

        # Use start_at_zero=False: distance coordinates start at half the distance
        selected = voxelmodel_var.gst.select_along_lines(
            lines, distance=0.4, start_at_zero=False
        )
        assert selected.sizes == {"z": 5, "line": 3, "distance": 7}
        assert_array_almost_equal(
            selected["distance"], [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6]
        )
        assert_array_equal(
            selected["x"],
            [
                [0.5, 1.5, 1.5, 1.5, 2.5, 2.5, np.nan],
                [np.nan, np.nan, 2.5, 3.5, 3.5, 3.5, 3.5],
                [3.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )
        assert_array_equal(
            selected["y"],
            [
                [1.5, 1.5, 1.5, 1.5, 2.5, 2.5, np.nan],
                [np.nan, np.nan, 0.5, 0.5, 0.5, 1.5, 1.5],
                [1.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

        # Use start_at_zero=False and drop=False
        selected = voxelmodel_var.gst.select_along_lines(
            lines, distance=0.4, drop=False, start_at_zero=False
        )
        assert selected.sizes == {"z": 5, "line": 4, "distance": 8}
        assert_array_almost_equal(
            selected["distance"], [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
        )
        assert_array_equal(
            selected["x"],
            [
                [0.5, 1.5, 1.5, 1.5, 2.5, 2.5, np.nan, np.nan],
                [np.nan, np.nan, 2.5, 3.5, 3.5, 3.5, 3.5, np.nan],
                [3.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )
        assert_array_equal(
            selected["y"],
            [
                [1.5, 1.5, 1.5, 1.5, 2.5, 2.5, np.nan, np.nan],
                [np.nan, np.nan, 0.5, 0.5, 0.5, 1.5, 1.5, np.nan],
                [1.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

        # Don't specify a distance, should fall back to using the model x-resolution as distance
        selected = voxelmodel_var.gst.select_along_lines(lines)
        assert selected.sizes == {"z": 5, "line": 3, "distance": 3}
        assert_array_almost_equal(selected["distance"], [0.0, 1.0, 2.0])
        assert_array_equal(selected["line"], [0, 1, 2])

        selected = layermodel_var.gst.select_along_lines(lines, start_at_zero=False)
        assert selected.sizes == {"layer": 4, "line": 2, "distance": 3}
        assert_array_almost_equal(selected["distance"], [0.5, 1.5, 2.5])
        assert_array_equal(
            selected["line"], [0, 1]
        )  # Line 3 falls outside the extent now

        selected = layermodel_var.gst.select_along_lines(
            lines, start_at_zero=False, drop=False
        )
        assert selected.sizes == {"layer": 4, "line": 4, "distance": 3}

    @pytest.mark.unittest
    def test_select_along_lines_wgs_coords(self, model_wgs, lines_wgs):
        selected = model_wgs.gst.select_along_lines(lines_wgs, distance=20)
        assert selected.sizes == {"z": 5, "line": 1, "distance": 5}
        assert_array_almost_equal(selected["distance"], [0.0, 20.0, 40.0, 60.0, 80.0])
        assert_array_equal(selected["line"], [0])
        assert_array_almost_equal(
            selected["x"],
            [[4.72694039, 4.72694039, 4.72694039, 4.72694039, 4.72694039]],
        )
        assert_array_almost_equal(
            selected["y"],
            [[52.3774435, 52.3774435, 52.3774435, 52.3774435, 52.3774435]],
        )

        selected = model_wgs.gst.select_along_lines(
            lines_wgs, distance=20, start_at_zero=False
        )
        assert selected.sizes == {"z": 5, "line": 1, "distance": 5}
        assert_array_almost_equal(selected["distance"], [10.0, 30.0, 50.0, 70.0, 90.0])

        selected = model_wgs.gst.select_along_lines(lines_wgs, drop=False)
        assert selected.sizes == {"z": 5, "line": 2, "distance": 12}
        assert_array_almost_equal(
            selected["distance"],
            [
                0.0,
                99.947068,
                199.894136,
                299.841203,
                399.788271,
                499.735339,
                599.682407,
                699.629475,
                799.576542,
                899.52361,
                999.470678,
                1099.417746,
            ],
        )

        # Check if the other way around works correctly too
        selected = model_wgs.gst.select_along_lines(
            lines_wgs.to_crs(28992), crs=28992, distance=20
        )
        assert selected.sizes == {"z": 5, "line": 1, "distance": 5}
        assert_array_almost_equal(selected["distance"], [0.0, 20.0, 40.0, 60.0, 80.0])
        assert_array_equal(selected["line"], [0])
        assert_array_almost_equal(
            selected["x"],
            [[4.72694039, 4.72694039, 4.72694039, 4.72694039, 4.72694039]],
        )
        assert_array_almost_equal(
            selected["y"],
            [[52.3774435, 52.3774435, 52.3774435, 52.3774435, 52.3774435]],
        )

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
    def test_slice_depth_interval(self, voxelmodel_var, layermodel_var, depth):
        """
        All behaviour for upper, lower, how, update_top_bottom and drop keyword arguments is
        tested in the module tests/models/test_voxelmodels.py and tests/models/test_layermodels.py.

        """
        sliced = voxelmodel_var.gst.slice_depth_interval(upper=depth, lower=depth - 1.2)
        assert isinstance(sliced, xr.DataArray)
        assert sliced.sizes == {"y": 4, "x": 4, "z": 4}
        assert_array_equal(sliced["z"], [-1.75, -1.25, -0.75, -0.25])

        sliced = layermodel_var.gst.slice_depth_interval(upper=depth, lower=depth - 1.2)
        assert isinstance(sliced, xr.DataArray)
        assert sliced.sizes == {"y": 4, "x": 4, "layer": 3}
        assert_array_equal(sliced["layer"], ["B", "C", "D"])
