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
