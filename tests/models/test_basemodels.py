import re

import numpy as np
import pytest
import pyvista as pv
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.models.basemodels import VoxelModel


@pytest.fixture
def point_shapefile(point_header, tmp_path):
    shapefile = tmp_path / "point_shapefile.shp"
    point_header.to_file(shapefile)
    return shapefile


@pytest.fixture
def point_parquet(point_header, tmp_path):
    parquet = tmp_path / "point_shapefile.geoparquet"
    point_header.to_parquet(parquet)
    return parquet


@pytest.fixture
def voxelmodel_netcdf(xarray_dataset, tmp_path):
    outfile = tmp_path / "voxelmodel.nc"
    xarray_dataset.to_netcdf(outfile)
    return outfile


@pytest.fixture
def simple_voxelmodel(xarray_dataset):
    col = xarray_dataset.isel(x=[1, 2], y=[1])
    return VoxelModel(col)


@pytest.fixture
def depth_mask(voxelmodel):
    return xr.DataArray(
        [
            [-0.5, -0.5, -0.8, -0.7],
            [-0.8, -0.8, -0.8, -0.8],
            [-0.7, -0.7, -0.7, -0.7],
            [-1.0, -1.0, -0.6, -0.6],
        ],
        coords={"y": voxelmodel["y"], "x": voxelmodel["x"]},
        dims=("y", "x"),
    )


class TestVoxelModel:
    @pytest.mark.unittest
    def test_from_netcdf(self, voxelmodel_netcdf):
        model = VoxelModel.from_netcdf(voxelmodel_netcdf)
        assert isinstance(model, VoxelModel)

        model = VoxelModel.from_netcdf(
            voxelmodel_netcdf, data_vars=["strat"], bbox=(1, 1, 3, 3), lazy=False
        )
        assert isinstance(model, VoxelModel)
        assert model.horizontal_bounds == (1, 1, 3, 3)
        assert list(model.variables) == ["strat"]

    @pytest.mark.unittest
    def test_initialize(self, xarray_dataset):
        model = VoxelModel(xarray_dataset)
        assert isinstance(model, VoxelModel)
        # TODO: Make sure an input DataArray becomes a Dataset which is needed by all methods.

    @pytest.mark.unittest
    def test_attributes(self, voxelmodel):
        assert voxelmodel.sizes == {"y": 4, "x": 4, "z": 5}
        assert voxelmodel.shape == (4, 4, 5)
        assert voxelmodel.resolution == (1, 1, 0.5)
        assert voxelmodel.horizontal_bounds == (0, 0, 4, 4)
        assert voxelmodel.vertical_bounds == (-2.5, 0)
        assert voxelmodel.crs == 28992
        assert_array_equal(voxelmodel.variables, ["strat", "lith"])
        assert voxelmodel.xmin == 0
        assert voxelmodel.ymin == 0
        assert voxelmodel.xmax == 4
        assert voxelmodel.ymax == 4
        assert voxelmodel.zmin == -2.5
        assert voxelmodel.zmax == 0

    @pytest.mark.unittest
    def test_sel(self, voxelmodel):
        ## Select exact coordinates
        selected = voxelmodel.sel(x=[1.5, 2.5])
        assert isinstance(selected, VoxelModel)
        assert selected.shape == (4, 2, 5)

        ## Other selections
        selected = voxelmodel.sel(x=[1.7, 2.3], method="nearest")
        assert selected.shape == (4, 2, 5)
        assert_array_equal(selected["x"], [1.5, 2.5])

        selected = voxelmodel.sel(x=slice(0.1, 2.5))
        assert selected.shape == (4, 3, 5)
        assert_array_equal(selected["x"], [0.5, 1.5, 2.5])

    @pytest.mark.unittest
    def test_isel(self, voxelmodel):
        selected = voxelmodel.isel(x=[0, 2])
        assert isinstance(selected, VoxelModel)
        assert selected.shape == (4, 2, 5)
        assert_array_equal(selected["x"], [0.5, 2.5])

        selected = voxelmodel.isel(x=slice(0, 2))
        assert selected.shape == (4, 2, 5)
        assert_array_equal(selected["x"], [0.5, 1.5])

    @pytest.mark.unittest
    def test_select_with_points(self, voxelmodel, borehole_collection):
        select = voxelmodel.select_with_points(borehole_collection.header)
        assert isinstance(select, xr.Dataset)
        assert select.sizes == {"idx": 4, "z": 5}
        assert_array_equal(select["idx"], [0, 1, 2, 4])
        assert_array_equal(select.data_vars, ["strat", "lith"])

    @pytest.mark.unittest
    @pytest.mark.parametrize("file", ["point_shapefile", "point_parquet"])
    def test_select_with_points_from_file(self, voxelmodel, file, request):
        file = request.getfixturevalue(file)
        select = voxelmodel.select_with_points(file)
        assert isinstance(select, xr.Dataset)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, voxelmodel, depth_mask):
        # Test selection where the upper and lower bounds cut through cells
        sliced = voxelmodel.slice_depth_interval(upper=-0.4, lower=-1.6)
        assert isinstance(sliced, VoxelModel)
        assert sliced.shape == (4, 4, 4)
        assert_array_equal(
            sliced["strat"],
            [
                [
                    [2.0, 2.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, np.nan],
                    [1.0, 1.0, 1.0, np.nan],
                    [2.0, 1.0, 1.0, np.nan],
                ],
                [
                    [2.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, np.nan],
                    [1.0, 2.0, 1.0, np.nan],
                ],
                [
                    [2.0, 2.0, 1.0, np.nan],
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, np.nan],
                ],
                [
                    [2.0, 2.0, 1.0, np.nan],
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, np.nan],
                    [2.0, 2.0, 1.0, 1.0],
                ],
            ],
        )
        assert_array_equal(
            sliced["lith"],
            (
                [
                    [
                        [3.0, 2.0, 1.0, 1.0],
                        [3.0, 1.0, 1.0, np.nan],
                        [1.0, 1.0, 1.0, np.nan],
                        [2.0, 1.0, 1.0, np.nan],
                    ],
                    [
                        [2.0, 1.0, 1.0, 1.0],
                        [2.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, np.nan],
                        [3.0, 2.0, 1.0, np.nan],
                    ],
                    [
                        [3.0, 2.0, 1.0, np.nan],
                        [1.0, 1.0, 3.0, 3.0],
                        [2.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 1.0, np.nan],
                    ],
                    [
                        [2.0, 2.0, 1.0, np.nan],
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 1.0, 3.0, np.nan],
                        [2.0, 2.0, 1.0, 1.0],
                    ],
                ]
            ),
        )

        # Test with upper and lower bounds at cell boundaries
        sliced = voxelmodel.slice_depth_interval(upper=-0.5, lower=-1.5)
        assert sliced.shape == (4, 4, 2)
        assert_array_equal(
            sliced["strat"],
            [
                [[2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                [[2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                [[2.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            ],
        )

        # Test using a depth grids as mask
        sliced = voxelmodel.slice_depth_interval(upper=depth_mask, lower=depth_mask - 1)
        assert_array_equal(
            sliced["strat"],
            [
                [
                    [np.nan, 2.0, 1.0],
                    [np.nan, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                ],
                [
                    [2.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0],
                ],
                [
                    [2.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                ],
                [
                    [2.0, 2.0, np.nan],
                    [1.0, 1.0, np.nan],
                    [2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                ],
            ],
        )
        assert_array_equal(
            sliced["lith"],
            [
                [
                    [np.nan, 2.0, 1.0],
                    [np.nan, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                ],
                [
                    [2.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 2.0, 1.0],
                ],
                [
                    [3.0, 2.0, 1.0],
                    [1.0, 1.0, 3.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                ],
                [
                    [2.0, 2.0, np.nan],
                    [1.0, 1.0, np.nan],
                    [2.0, 1.0, 3.0],
                    [2.0, 2.0, 1.0],
                ],
            ],
        )

        # Test with drop=False --> keep original shape of sliced but set values to NaN
        sliced = voxelmodel.slice_depth_interval(upper=-0.4, lower=-1.6, drop=False)
        assert sliced.shape == voxelmodel.shape
        assert_array_equal(
            sliced["strat"],
            [
                [
                    [np.nan, 2.0, 2.0, 1.0, 1.0],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                ],
                [
                    [np.nan, 2.0, 1.0, 1.0, 1.0],
                    [np.nan, 2.0, 1.0, 1.0, 1.0],
                    [np.nan, 1.0, 1.0, 1.0, np.nan],
                    [np.nan, 1.0, 2.0, 1.0, np.nan],
                ],
                [
                    [np.nan, 2.0, 2.0, 1.0, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, 1.0],
                    [np.nan, 2.0, 1.0, 1.0, 1.0],
                    [np.nan, 2.0, 2.0, 1.0, np.nan],
                ],
                [
                    [np.nan, 2.0, 2.0, 1.0, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, 1.0],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 2.0, 1.0, 1.0],
                ],
            ],
        )

        sliced = voxelmodel.slice_depth_interval(depth_mask, depth_mask - 1, drop=False)
        assert sliced.shape == voxelmodel.shape
        assert_array_equal(
            sliced["strat"],
            [
                [
                    [np.nan, np.nan, 2.0, 1.0, np.nan],
                    [np.nan, np.nan, 1.0, 1.0, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                ],
                [
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, np.nan],
                    [np.nan, 1.0, 2.0, 1.0, np.nan],
                ],
                [
                    [np.nan, 2.0, 2.0, 1.0, np.nan],
                    [np.nan, 1.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 2.0, 1.0, np.nan],
                ],
                [
                    [np.nan, 2.0, 2.0, np.nan, np.nan],
                    [np.nan, 1.0, 1.0, np.nan, np.nan],
                    [np.nan, 2.0, 1.0, 1.0, np.nan],
                    [np.nan, 2.0, 2.0, 1.0, np.nan],
                ],
            ],
        )

        # Test error when upper is deeper than lower
        sliced = voxelmodel.slice_depth_interval(upper=-1.5, lower=-0.5)
        assert sliced.shape == (4, 4, 0)

        sliced = voxelmodel.slice_depth_interval(upper=-0.5, lower=depth_mask)
        assert_array_equal(
            sliced["strat"],
            [
                [[np.nan], [np.nan], [1.0], [1.0]],
                [[1.0], [1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0], [1.0]],
            ],
        )

        # Test with 1D DataArray inputs
        da_1d = xr.DataArray([-1.0, -1.5, -2.0, -2.5], dims=["x"])
        sliced = voxelmodel.slice_depth_interval(upper=da_1d, lower=da_1d - 1)
        assert_array_equal(
            sliced["strat"],
            [
                [[np.nan, 2.0, 2.0], [2.0, 2.0, np.nan], [2.0, np.nan, np.nan]],
                [[np.nan, 2.0, 1.0], [2.0, 2.0, np.nan], [2.0, np.nan, np.nan]],
                [[np.nan, 2.0, 2.0], [2.0, 1.0, np.nan], [2.0, np.nan, np.nan]],
                [[np.nan, 2.0, 2.0], [2.0, 1.0, np.nan], [2.0, np.nan, np.nan]],
            ],
        )

        with pytest.raises(
            TypeError, match="Input for 'upper' must be int, float or xr.DataArray"
        ):
            voxelmodel.slice_depth_interval(upper="invalid", lower=-1.5)

        with pytest.raises(
            TypeError, match="Input for 'lower' must be int, float or xr.DataArray"
        ):
            voxelmodel.slice_depth_interval(upper=-1.5, lower="invalid")

    @pytest.mark.xfail(reason="Not implemented yet.")
    @pytest.mark.parametrize(
        "how, result",
        [("overlap", 2), ("majority", 2), ("inner", 2)],
        ids=["overlap", "majority", "inner"],
    )
    def test_slice_depth_interval_how(self, simple_voxelmodel, how, result):
        sliced = simple_voxelmodel.slice_depth_interval(upper=-0.4, lower=-1.6, how=how)
        assert isinstance(sliced, VoxelModel)

    @pytest.mark.unittest
    def test_thickness_map_single_condition(self, voxelmodel):
        thickness_lith1 = voxelmodel.get_thickness(voxelmodel["lith"] == 1)
        thickness_lith2 = voxelmodel.get_thickness(voxelmodel["lith"] == 2)
        thickness_lith3 = voxelmodel.get_thickness(voxelmodel["lith"] == 3)
        assert isinstance(thickness_lith1, xr.DataArray)
        assert isinstance(thickness_lith2, xr.DataArray)
        assert isinstance(thickness_lith3, xr.DataArray)
        assert_array_equal(
            thickness_lith1,
            np.array(
                [
                    [1.0, 1.0, 1.5, 1.0],
                    [1.5, 1.5, 1.5, 0.5],
                    [0.5, 1.0, 1.5, 0.5],
                    [0.5, 2.0, 0.5, 1.0],
                ]
            ),
        )
        assert_array_equal(
            thickness_lith2,
            np.array(
                [
                    [1.0, 0.5, 0.5, 0.5],
                    [1.0, 1.0, 0.5, 1.0],
                    [1.0, 0.5, 1.0, 1.0],
                    [1.5, 0.5, 1.0, 1.0],
                ]
            ),
        )
        assert_array_equal(
            thickness_lith3,
            np.array(
                [
                    [0.5, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.0, 0.5],
                    [0.5, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5],
                ]
            ),
        )

    @pytest.mark.unittest
    def test_thickness_map_extra_conditions(self, voxelmodel):
        thickness_lith2_strat2 = voxelmodel.get_thickness(
            ((voxelmodel["lith"] == 2) & (voxelmodel["strat"] == 2))
        )
        thickness_lith2_zrange = voxelmodel.get_thickness(
            voxelmodel["lith"] == 2, depth_range=(-1.5, -0.5)
        )
        assert isinstance(thickness_lith2_strat2, xr.DataArray)
        assert isinstance(thickness_lith2_zrange, xr.DataArray)
        assert_array_equal(
            thickness_lith2_strat2,
            np.array(
                [
                    [1.0, 0.5, 0.5, 0.5],
                    [1.0, 1.0, 0.5, 1.0],
                    [1.0, 0.5, 1.0, 1.0],
                    [1.5, 0.5, 1.0, 1.0],
                ]
            ),
        )
        assert_array_equal(
            thickness_lith2_zrange,
            np.array(
                [
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.0, 0.5],
                ]
            ),
        )

    @pytest.mark.unittest
    def test_most_common(self, voxelmodel):
        result = voxelmodel.most_common("lith")
        assert isinstance(result, xr.Dataset)
        assert_array_equal(
            result["most_common"],
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 2.0],
                [2.0, 1.0, 1.0, 2.0],
                [2.0, 1.0, 2.0, 1.0],
            ],
        )
        assert_array_almost_equal(
            result["thickness_most_common"],
            [
                [1.0, 1.0, 1.5, 1.0],
                [1.5, 1.5, 1.5, 1.0],
                [1.0, 1.0, 1.5, 1.0],
                [1.5, 2.0, 1.0, 1.0],
            ],
        )

    @pytest.mark.unittest
    def test_value_counts(self, voxelmodel):
        result = voxelmodel.value_counts("lith")
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("lith",)
        assert_array_equal(result, [34, 27, 9])
        assert_array_equal(result["lith"], [1, 2, 3])

        result = voxelmodel.value_counts("lith", normalize=True)
        assert_array_almost_equal(result, [0.48571429, 0.38571429, 0.12857143])

        result = voxelmodel.value_counts("strat")
        assert_array_equal(result, [38, 32])
        assert_array_equal(result["strat"], [1, 2])

        result = voxelmodel.value_counts("lith", dim="z")
        assert_array_equal(
            result,
            [
                [[2, 2, 3, 2], [3, 3, 3, 1], [1, 2, 3, 1], [1, 4, 1, 2]],
                [[2, 1, 1, 1], [2, 2, 1, 2], [2, 1, 2, 2], [3, 1, 2, 2]],
                [[1, 1, 0, 1], [0, 0, 0, 1], [1, 2, 0, 0], [0, 0, 1, 1]],
            ],
        )
        assert_array_equal(result["lith"], [1, 2, 3])

        result = voxelmodel.value_counts("lith", dim="z", normalize=True)
        assert_array_almost_equal(
            result,
            [
                [
                    [0.4, 0.5, 0.75, 0.5],
                    [0.6, 0.6, 0.75, 0.25],
                    [0.25, 0.4, 0.6, 0.33333333],
                    [0.25, 0.8, 0.25, 0.4],
                ],
                [
                    [0.4, 0.25, 0.25, 0.25],
                    [0.4, 0.4, 0.25, 0.5],
                    [0.5, 0.2, 0.4, 0.66666667],
                    [0.75, 0.2, 0.5, 0.4],
                ],
                [
                    [0.2, 0.25, 0.0, 0.25],
                    [0.0, 0.0, 0.0, 0.25],
                    [0.25, 0.4, 0.0, 0.0],
                    [0.0, 0.0, 0.25, 0.2],
                ],
            ],
        )

        result = voxelmodel.value_counts("strat", dim="x")
        assert_array_equal(
            result,
            [
                [[0, 1, 3, 4, 1], [0, 2, 3, 4, 2], [0, 1, 2, 4, 2], [0, 1, 2, 4, 2]],
                [[4, 3, 1, 0, 0], [4, 2, 1, 0, 0], [3, 3, 2, 0, 0], [4, 3, 2, 0, 0]],
            ],
        )
        assert_array_equal(result["strat"], [1, 2])

        result = voxelmodel.value_counts("strat", dim="y")
        assert_array_equal(
            result,
            [
                [[0, 0, 1, 4, 2], [0, 2, 4, 4, 3], [0, 2, 4, 4, 1], [0, 1, 1, 4, 1]],
                [[4, 4, 3, 0, 0], [4, 2, 0, 0, 0], [4, 2, 0, 0, 0], [3, 3, 3, 0, 0]],
            ],
        )

    @pytest.mark.unittest
    def test_to_pyvista_structured(self, voxelmodel):
        vms_single_var = voxelmodel.to_pyvista_grid(data_vars=["strat"])
        assert isinstance(vms_single_var, pv.ImageData)
        assert vms_single_var.n_points == 150
        assert vms_single_var.n_cells == 80
        assert vms_single_var.n_arrays == 1

        vms_multi_var = voxelmodel.to_pyvista_grid()
        assert isinstance(vms_multi_var, pv.ImageData)
        assert vms_multi_var.n_points == 150
        assert vms_multi_var.n_cells == 80
        assert vms_multi_var.n_arrays == 2

    @pytest.mark.xfail(
        reason="Unclear why a different number of arrays is returned instead of 1."
    )
    @pytest.mark.unittest
    def test_to_pyvista_unstructured(self, voxelmodel):
        vmu_single_var = voxelmodel.to_pyvista_grid(
            data_vars=["strat"], structured=False
        )
        assert isinstance(vmu_single_var, pv.UnstructuredGrid)
        assert vmu_single_var.n_points == 560
        assert vmu_single_var.n_cells == 70
        assert vmu_single_var.n_arrays == 1  # <-- should be 1 but is 3?

        vmu_multi_var = voxelmodel.to_pyvista_grid(structured=False)
        assert isinstance(vmu_multi_var, pv.UnstructuredGrid)
        assert vmu_multi_var.n_points == 560
        assert vmu_multi_var.n_cells == 70
        assert vmu_multi_var.n_arrays == 2  # <-- should be 2 but is 4?

    @pytest.mark.xfail(
        reason=(
            "Fails due to same reason as test_to_pyvista_unstructured. Is the part where "
            "it fails (assert vmu_wrong_order.n_arrays == 2) even needed?"
        )
    )
    @pytest.mark.unittest
    def test_to_pyvista_unstructured_problematic_dims(self, voxelmodel):
        # Wrong order of dimensions leads to automatic transposing, not an error!

        # Why are the five line below in this test? The same happens in the test above.
        vmu_wrong_order = voxelmodel.to_pyvista_grid(structured=False)
        assert isinstance(vmu_wrong_order, pv.UnstructuredGrid)
        assert vmu_wrong_order.n_points == 560
        assert vmu_wrong_order.n_cells == 70
        assert vmu_wrong_order.n_arrays == 2

        # Missing z-dimension leads to an error and no file is created.
        voxelmodel.ds = voxelmodel.ds.drop_vars("z")
        with pytest.raises(Exception) as error_info:
            voxelmodel.to_pyvista_grid()
        assert error_info.errisinstance(ValueError)
        assert error_info.match(
            re.escape(
                "Dataset must contain 'z' dimension. Make sure that this "
                "spatial dimension exists in the dataset or if it has a different "
                "name use xarray.Dataset.rename() to rename the corresponding "
                "dimension to 'z'."
            )
        )
