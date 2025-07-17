import re

import numpy as np
import pytest
import pyvista as pv
import xarray as xr
from numpy.testing import assert_array_equal

from geost.models.basemodels import VoxelModel


@pytest.fixture
def point_shapefile(point_header_gdf, tmp_path):
    shapefile = tmp_path / "point_shapefile.shp"
    point_header_gdf.to_file(shapefile)
    return shapefile


@pytest.fixture
def point_parquet(point_header_gdf, tmp_path):
    parquet = tmp_path / "point_shapefile.geoparquet"
    point_header_gdf.to_parquet(parquet)
    return parquet


@pytest.fixture
def voxelmodel_netcdf(xarray_dataset, tmp_path):
    outfile = tmp_path / "voxelmodel.nc"
    xarray_dataset.to_netcdf(outfile)
    return outfile


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
        assert voxelmodel.sizes == {"y": 4, "x": 4, "z": 4}
        assert voxelmodel.shape == (4, 4, 4)
        assert voxelmodel.resolution == (1, 1, 0.5)
        assert voxelmodel.horizontal_bounds == (0, 0, 4, 4)
        assert voxelmodel.vertical_bounds == (-2, 0)
        assert voxelmodel.crs == 28992
        assert_array_equal(voxelmodel.variables, ["strat", "lith"])
        assert voxelmodel.xmin == 0
        assert voxelmodel.ymin == 0
        assert voxelmodel.xmax == 4
        assert voxelmodel.ymax == 4
        assert voxelmodel.zmin == -2
        assert voxelmodel.zmax == 0

    @pytest.mark.unittest
    def test_sel(self, voxelmodel):
        ## Select exact coordinates
        selected = voxelmodel.sel(x=[1.5, 2.5])
        assert isinstance(selected, VoxelModel)
        assert selected.shape == (4, 2, 4)

        ## Other selections
        selected = voxelmodel.sel(x=[1.7, 2.3], method="nearest")
        assert selected.shape == (4, 2, 4)
        assert_array_equal(selected["x"], [1.5, 2.5])

        selected = voxelmodel.sel(x=slice(0.1, 2.5))
        assert selected.shape == (4, 3, 4)
        assert_array_equal(selected["x"], [0.5, 1.5, 2.5])

    @pytest.mark.unittest
    def test_isel(self, voxelmodel):
        selected = voxelmodel.isel(x=[0, 2])
        assert isinstance(selected, VoxelModel)
        assert selected.shape == (4, 2, 4)
        assert_array_equal(selected["x"], [0.5, 2.5])

        selected = voxelmodel.isel(x=slice(0, 2))
        assert selected.shape == (4, 2, 4)
        assert_array_equal(selected["x"], [0.5, 1.5])

    @pytest.mark.unittest
    def test_select_with_points(self, voxelmodel, borehole_collection):
        points = borehole_collection.header.gdf
        select = voxelmodel.select_with_points(points)
        assert isinstance(select, xr.Dataset)
        assert select.sizes == {"idx": 4, "z": 4}
        assert_array_equal(select["idx"], [0, 1, 2, 4])
        assert_array_equal(select.data_vars, ["strat", "lith"])

    @pytest.mark.unittest
    @pytest.mark.parametrize("file", ["point_shapefile", "point_parquet"])
    def test_select_with_points_from_file(self, voxelmodel, file, request):
        file = request.getfixturevalue(file)
        select = voxelmodel.select_with_points(file)
        assert isinstance(select, xr.Dataset)

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
                    [0.5, 1.0, 1.5, 1.0],
                    [1.0, 1.0, 1.5, 0.5],
                    [0.5, 1.0, 1.0, 0.5],
                    [0.5, 1.5, 0.5, 0.5],
                ]
            ),
        )
        assert_array_equal(
            thickness_lith2,
            np.array(
                [
                    [1.0, 0.5, 0.5, 0.5],
                    [1.0, 1.0, 0.5, 1.0],
                    [1.0, 0.5, 1.0, 1.5],
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
                    [0.5, 0.5, 0.0, 0.0],
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
            voxelmodel["lith"] == 2, depth_range=(-1, 0)
        )
        assert isinstance(thickness_lith2_strat2, xr.DataArray)
        assert isinstance(thickness_lith2_zrange, xr.DataArray)
        assert_array_equal(
            thickness_lith2_strat2,
            np.array(
                [
                    [1.0, 0.5, 0.5, 0.5],
                    [1.0, 1.0, 0.5, 1.0],
                    [1.0, 0.5, 1.0, 1.5],
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
    def test_to_pyvista_structured(self, voxelmodel):
        vms_single_var = voxelmodel.to_pyvista_grid(data_vars=["strat"])
        vms_multi_var = voxelmodel.to_pyvista_grid()
        assert isinstance(vms_single_var, pv.ImageData)
        assert isinstance(vms_multi_var, pv.ImageData)
        assert vms_single_var.n_points == 125
        assert vms_single_var.n_cells == 64
        assert vms_single_var.n_arrays == 1
        assert vms_multi_var.n_points == 125
        assert vms_multi_var.n_cells == 64
        assert vms_multi_var.n_arrays == 2

    @pytest.mark.unittest
    def test_to_pyvista_unstructured(self, voxelmodel):
        vmu_single_var = voxelmodel.to_pyvista_grid(
            data_vars=["strat"], structured=False
        )
        vmu_multi_var = voxelmodel.to_pyvista_grid(structured=False)
        assert isinstance(vmu_single_var, pv.UnstructuredGrid)
        assert isinstance(vmu_multi_var, pv.UnstructuredGrid)
        assert vmu_single_var.n_points == 512
        assert vmu_single_var.n_cells == 64
        assert vmu_single_var.n_arrays == 1
        assert vmu_multi_var.n_points == 512
        assert vmu_multi_var.n_cells == 64
        assert vmu_multi_var.n_arrays == 2

    @pytest.mark.unittest
    def test_to_pyvista_unstructured_problematic_dims(self, voxelmodel):
        # Wrong order of dimensions leads to automatic transposing, not an error!
        vmu_wrong_order = voxelmodel.to_pyvista_grid(structured=False)
        assert isinstance(vmu_wrong_order, pv.UnstructuredGrid)
        assert vmu_wrong_order.n_points == 512
        assert vmu_wrong_order.n_cells == 64
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
