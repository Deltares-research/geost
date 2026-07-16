import geopandas as gpd
import pytest
import shapely


@pytest.fixture
def points():
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.8, 2.4, 1.0], [0.8, 2.4, 0.5]), crs="EPSG:28992"
    )  # The third point is exactly on the edge of bordering cells


@pytest.fixture
def lines():
    return gpd.GeoDataFrame(
        geometry=[
            shapely.LineString([(0.8, 0.9), (2.4, 2.5)]),
            shapely.LineString([(2.9, 0.1), (3.5, 1.6)]),
        ],
        crs="EPSG:28992",
    )


@pytest.fixture
def polygons():
    return gpd.GeoDataFrame(
        geometry=[shapely.box(0.4, 0.4, 2.4, 2.4), shapely.box(2.8, 1.8, 3.2, 2.2)],
        crs="EPSG:28992",
    )
