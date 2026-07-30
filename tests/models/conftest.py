import geopandas as gpd
import pytest
import shapely


@pytest.fixture
def points():
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.8, 2.4, 1.0, 4.5], [0.8, 2.4, 0.5, 4.5]),
        crs="EPSG:28992",
    )  # The third point is exactly on the edge of bordering cells, the fourth point is outside the model extent


@pytest.fixture
def lines():
    """
    line 1: fully within model extent
    line 2: begins outside model extent but is largely within
    line 3: begins inside model extent but is largely outside, also the longest line
    line 4: fully outside model extent

    """
    return gpd.GeoDataFrame(
        geometry=[
            shapely.LineString([(0.8, 0.9), (2.4, 2.5)]),
            shapely.LineString([(2.9, -1), (2.9, 0.1), (3.5, 1.6)]),
            shapely.LineString([(3.7, 1.4), (6.7, 1.4)]),
            shapely.LineString([(4.5, 4.5), (5.0, 5.0)]),
        ],
        crs="EPSG:28992",
    )


@pytest.fixture
def polygons():
    return gpd.GeoDataFrame(
        geometry=[
            shapely.box(0.4, 0.4, 2.4, 2.4),
            shapely.box(2.8, 1.8, 3.2, 2.2),
            shapely.box(4.5, 4.5, 5.0, 5.0),  # Outside the model extent
        ],
        crs="EPSG:28992",
    )
