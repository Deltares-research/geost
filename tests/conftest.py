from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geost import read_sst_cores
from geost.data_objects import LayeredData


@pytest.fixture
def borehole_file():
    return Path(__file__).parent / "data" / "test_boreholes.parquet"


@pytest.fixture
def borehole_collection(borehole_file):
    borehole_collection = read_sst_cores(borehole_file)
    return borehole_collection


@pytest.fixture
def borehole_data(borehole_file):
    df = pd.read_parquet(borehole_file)
    return LayeredData(df)


# Fixtures for header testing
@pytest.fixture
def point_header_gdf():
    """
    Creates a synthetic header geodataframe for testing
    """
    x_coors = [1, 2, 3, 4, 5]
    y_coors = [1, 2, 3, 4, 5]
    coordinates = np.array([(x, y) for x, y in product(x_coors, y_coors)])
    nrs = ["nr" + str(i + 1) for i in range(len(coordinates))]
    mvs = np.arange(1, 26)
    ends = np.arange(-1, -26, -1)
    geometries = [Point(c) for c in coordinates]
    gdf = gpd.GeoDataFrame(
        {
            "nr": nrs,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "mv": mvs,
            "end": ends,
        },
        geometry=geometries,
    )
    return gdf
