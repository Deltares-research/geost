import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from numpy.testing import assert_allclose, assert_array_almost_equal

from geost.utils import spatial
from geost.utils.conversion import dataframe_to_geodataframe


@pytest.fixture
def two_lines():
    return gpd.GeoDataFrame(
        geometry=[
            shapely.LineString([(0.8, 0.9), (2.4, 2.5)]),
            shapely.LineString([(2.9, 0.1), (3.5, 1.6)]),
        ],
        crs="EPSG:28992",
    )


class TestSpatialUtils:
    @pytest.fixture
    def raster(self):
        x_coors = np.arange(1, 4)
        y_coors = np.arange(4, 1, -1)
        data = np.ones((3, 3))
        array = xr.DataArray(data, {"x": x_coors, "y": y_coors})
        return array

    @pytest.fixture
    def invalid_raster(self):
        x_coors = np.arange(1, 4)
        y_coors = np.arange(4, 1, -1)
        data = np.ones((3, 3))
        array = xr.DataArray(data, {"invalid_x": x_coors, "invalid_y": y_coors})
        return array

    @pytest.fixture
    def dataframe_with_coordinates(self):
        dataframe = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        return dataframe

    @pytest.mark.unittest
    def test_dataframe_to_geodataframe(self, dataframe_with_coordinates):
        gdf = dataframe_to_geodataframe(dataframe_with_coordinates, crs=28992)
        assert isinstance(gdf["geometry"].dtype, gpd.array.GeometryDtype)

    # @pytest.mark.unittest
    # def test_check_geometry_instance(self, point_header):
    #     point_header.to_parquet("temp_file.geoparquet")
    #     point_header.to_file("temp_file.gpkg")
    #     point_header_geoparquet = "temp_file.geoparquet"
    #     point_header_geopackage = "temp_file.gpkg"
    #     gdf_gdf = spatial.check_geometry_instance(point_header)
    #     gdf_geoparquet_gdf = spatial.check_geometry_instance(
    #         point_header_geoparquet
    #     )
    #     gdf_geopackage_gdf = spatial.check_geometry_instance(
    #         point_header_geopackage
    #     )
    #     assert isinstance(gdf_gdf, gpd.GeoDataFrame)
    #     assert isinstance(gdf_geoparquet_gdf, gpd.GeoDataFrame)
    #     assert isinstance(gdf_geopackage_gdf, gpd.GeoDataFrame)
    #     Path("temp_file.geoparquet").unlink()
    #     Path("temp_file.gpkg").unlink()

    @pytest.mark.unittest
    def test_check_and_coerce_crs(self, point_header):
        with pytest.warns(UserWarning):
            referenced_gdf = spatial.check_and_coerce_crs(point_header, 28992)
            converted_referenced_gdf = spatial.check_and_coerce_crs(
                referenced_gdf, 32631
            )
        assert referenced_gdf.crs == "epsg:28992"
        assert converted_referenced_gdf.crs == "epsg:32631"

    @pytest.mark.unittest
    def test_get_raster_values(self, raster, dataframe_with_coordinates):
        x = dataframe_with_coordinates["x"].values
        y = dataframe_with_coordinates["y"].values
        raster_values = spatial.get_raster_values(x, y, raster)
        assert_allclose(raster_values, np.array([np.nan, 1.0, 1.0, np.nan, np.nan]))

    @pytest.mark.unittest
    def test_get_invalid_raster_values(
        self, invalid_raster, dataframe_with_coordinates
    ):
        # In this case the used raster does not have the correct labels for x and y
        # coordinates, which should raise an error.
        x = dataframe_with_coordinates["x"].values
        y = dataframe_with_coordinates["y"].values
        with pytest.raises(Exception) as error_info:
            spatial.get_raster_values(x, y, invalid_raster)
        assert error_info.errisinstance(TypeError)
        assert error_info.match(
            "The xr.DataArray to sample from does not have the "
            + "required 'x' and 'y' dimensions"
        )


@pytest.fixture
def surface_points():
    return gpd.GeoSeries(
        gpd.points_from_xy([0.5, 1.5, 10.5], [0.5, 1.5, 10.5]),
        crs=28992,
    )


@pytest.mark.unittest
@pytest.mark.parametrize("value", [2.33, 0])
def test_transform_surface_level_constant(monkeypatch, surface_points, value):
    monkeypatch.setitem(
        spatial.VERTICAL_DATUM_CONVERSIONS, "EPSG:5709>EPSG:5710", value
    )
    result = spatial.transform_surface_level(surface_points, 28992, 5709, "EPSG:5710")
    assert_allclose(result, np.full(3, value))


@pytest.mark.unittest
def test_transform_surface_level_grid(monkeypatch, surface_points):
    grid = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    ).rio.write_crs(28992)
    monkeypatch.setitem(spatial.VERTICAL_DATUM_CONVERSIONS, "EPSG:5709>EPSG:5861", grid)
    result = spatial.transform_surface_level(surface_points, 28992, 5709, 5861)
    assert_allclose(result, [3.0, 2.0, np.nan])


@pytest.mark.unittest
def test_transform_surface_level_unknown_conversion(surface_points):
    with pytest.raises(ValueError, match="No conversion available"):
        spatial.transform_surface_level(surface_points, 28992, 5709, 5709)


@pytest.mark.unittest
def test_get_points_along_lines(two_lines):
    dist = 0.4
    distances = np.arange(0, two_lines.length.max(), dist)
    points = spatial.get_points_along_lines(two_lines, distances)
    assert isinstance(points, gpd.GeoDataFrame)
    assert len(points) == 11
    assert points.index.name == "line"
    assert_array_almost_equal(
        points["distance"], [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 0.0, 0.4, 0.8, 1.2, 1.6]
    )
    assert_array_almost_equal(
        points["geometry"].x,
        [
            0.8,
            1.08284271,
            1.36568542,
            1.64852814,
            1.93137085,
            2.21421356,
            2.9,
            3.04855627,
            3.19711254,
            3.34566881,
            3.49422508,
        ],
    )
    assert_array_almost_equal(
        points["geometry"].y,
        [
            0.9,
            1.18284271,
            1.46568542,
            1.74852814,
            2.03137085,
            2.31421356,
            0.1,
            0.47139068,
            0.84278135,
            1.21417203,
            1.58556271,
        ],
    )

    points = spatial.get_points_along_lines(two_lines, distances + 0.2)
    assert len(points) == 10
    assert_array_almost_equal(
        points["distance"], [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 0.2, 0.6, 1.0, 1.4]
    )
