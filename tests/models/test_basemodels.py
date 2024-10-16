import pytest
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


class TestVoxelModel:
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
        assert voxelmodel.vertical_bounds == (0, 2)
        assert voxelmodel.crs == 28992
        assert_array_equal(voxelmodel.variables, ["strat", "lith"])
        assert voxelmodel.xmin == 0
        assert voxelmodel.ymin == 0
        assert voxelmodel.xmax == 4
        assert voxelmodel.ymax == 4
        assert voxelmodel.zmin == 0
        assert voxelmodel.zmax == 2

    @pytest.mark.unittest
    def test_select(self, voxelmodel):
        ## Select exact coordinates
        selected = voxelmodel.select(x=[1.5, 2.5])
        assert isinstance(selected, VoxelModel)
        assert selected.shape == (4, 2, 4)

        ## Other selections
        selected = voxelmodel.select(x=[1.7, 2.3], method="nearest")
        assert selected.shape == (4, 2, 4)
        assert_array_equal(selected["x"], [1.5, 2.5])

        selected = voxelmodel.select(x=slice(0.1, 2.5))
        assert selected.shape == (4, 3, 4)
        assert_array_equal(selected["x"], [0.5, 1.5, 2.5])

    @pytest.mark.unittest
    def test_select_index(self, voxelmodel):
        selected = voxelmodel.select_index(x=[0, 2])
        assert isinstance(selected, VoxelModel)
        assert selected.shape == (4, 2, 4)
        assert_array_equal(selected["x"], [0.5, 2.5])

        selected = voxelmodel.select_index(x=slice(0, 2))
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
