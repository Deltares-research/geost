from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geost import read_sst_cores
from geost.new_base import LayeredData


def borehole_a():
    nlayers = 5
    top = [0, 0.8, 1.5, 2.5, 3.7]
    bottom = top[1:] + [4.2]
    mv = 0.2
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "A"),
            "x": np.full(nlayers, 2),
            "y": np.full(nlayers, 3),
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["K", "K", "Z", "Z", "K"],
        }
    )


def borehole_b():
    nlayers = 5
    top = [0, 0.6, 1.2, 2.5, 3.1]
    bottom = top[1:] + [3.9]
    mv = 0.3
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "B"),
            "x": np.full(nlayers, 1),
            "y": np.full(nlayers, 4),
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["K", "K", "V", "V", "K"],
        }
    )


def borehole_c():
    nlayers = 5
    top = [0, 1.4, 1.8, 2.9, 3.8]
    bottom = top[1:] + [5.5]
    mv = 0.25
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "C"),
            "x": np.full(nlayers, 4),
            "y": np.full(nlayers, 2),
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["K", "K", "K", "Z", "Z"],
        }
    )


def borehole_d():
    nlayers = 5
    top = [0, 0.5, 1.2, 1.8, 2.5]
    bottom = top[1:] + [3.0]
    mv = 0.1
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "D"),
            "x": np.full(nlayers, 3),
            "y": np.full(nlayers, 5),
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["K", "V", "K", "V", "Z"],
        }
    )


def borehole_e():
    nlayers = 5
    top = [0, 0.5, 1.2, 1.8, 2.5]
    bottom = top[1:] + [3.0]
    mv = -0.1
    end = mv - bottom[-1]
    return pd.DataFrame(
        {
            "nr": np.full(nlayers, "E"),
            "x": np.full(nlayers, 1),
            "y": np.full(nlayers, 1),
            "surface": np.full(nlayers, mv),
            "end": np.full(nlayers, end),
            "top": top,
            "bottom": bottom,
            "lith": ["Z", "Z", "Z", "Z", "Z"],
        }
    )


@pytest.fixture
def borehole_file():
    return Path(__file__).parent / "data" / "test_boreholes.parquet"


@pytest.fixture
def borehole_collection(borehole_file):
    borehole_collection = read_sst_cores(borehole_file)
    return borehole_collection


@pytest.fixture
def borehole_data():
    a = borehole_a()
    b = borehole_b()
    c = borehole_c()
    d = borehole_d()
    e = borehole_e()
    df = pd.concat([a, b, c, d, e], ignore_index=True)
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
            "surface": mvs,
            "end": ends,
        },
        geometry=geometries,
    )
    gdf.set_crs("epsg:28992", inplace=True)
    return gdf
