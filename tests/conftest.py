from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import geometry as gmt

from geost import read_borehole_table, read_nlog_cores
from geost.base import DiscreteData, LayeredData
from geost.models.basemodels import VoxelModel


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
def testdatadir():
    return Path(__file__).parent / "data"


@pytest.fixture
def borehole_file(testdatadir):
    return testdatadir / r"test_boreholes.parquet"


@pytest.fixture
def nlog_borehole_file(testdatadir):
    return testdatadir / r"test_nlog_stratstelsel_20230807.parquet"


@pytest.fixture
def borehole_collection(borehole_data):
    """
    Fixture containing a BoreholeCollection instance of synthetic borehole data.

    """
    borehole_collection = borehole_data.to_collection()
    return borehole_collection


@pytest.fixture
def nlog_borehole_collection(nlog_borehole_file):
    nlog_borehole_collection = read_nlog_cores(nlog_borehole_file)
    return nlog_borehole_collection


@pytest.fixture
def borehole_data():
    a = borehole_a()
    b = borehole_b()
    c = borehole_c()
    d = borehole_d()
    e = borehole_e()
    df = pd.concat([a, b, c, d, e], ignore_index=True)
    """
    Fixture containing a LayeredData instance of synthetic borehole data.

    """
    return LayeredData(df)


# Fixtures for header testing
@pytest.fixture
def point_header_gdf():
    """
    Creates a synthetic header geodataframe for testing
    """
    x_coors = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_coors = [1.0, 2.0, 3.0, 4.0, 5.0]
    coordinates = np.array([(x, y) for x, y in product(x_coors, y_coors)])
    nrs = ["nr" + str(i + 1) for i in range(len(coordinates))]
    mvs = np.arange(1, 26)
    ends = np.arange(-1, -26, -1)
    geometries = [gmt.Point(c) for c in coordinates]
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


def cpt_a():
    """
    Helper function for a synthetic CPT containing qs, fs and u2 "measurements".

    """
    depth = np.arange(10)
    surface = 2.1
    end = surface - depth.max()
    qc = [0.227, 0.279, 0.327, 0.354, 0.357, 0.354, 0.363, 0.447, 0.761, 1.481]
    fs = [0.010, 0.014, 0.019, 0.021, 0.022, 0.023, 0.026, 0.023, 0.022, 0.021]
    u2 = [0.018, 0.026, 0.035, 0.041, 0.047, 0.052, 0.058, 0.061, 0.057, 0.036]
    return pd.DataFrame(
        {
            "nr": np.repeat("a", 10),
            "x": np.repeat(1, 10),
            "y": np.repeat(1, 10),
            "surface": np.repeat(surface, 10),
            "end": np.repeat(end, 10),
            "depth": depth,
            "qc": qc,
            "fs": fs,
            "u2": u2,
        }
    )


def cpt_b():
    """
    Helper function for a synthetic CPT containing qs, fs and u2 "measurements".

    """
    depth = np.arange(10)
    surface = 0.8
    end = surface - depth.max()
    qc = [8.721, 12.733, 17.324, 17.036, 16.352, 15.781, 15.365, 15.509, 15.884, 15.982]
    fs = [0.061, 0.058, 0.055, 0.054, 0.052, 0.051, 0.052, 0.051, 0.051, 0.050]
    u2 = [0.218, 0.219, 0.221, 0.220, 0.219, 0.220, 0.221, 0.224, 0.224, 0.225]
    return pd.DataFrame(
        {
            "nr": np.repeat("b", 10),
            "x": np.repeat(2, 10),
            "y": np.repeat(2, 10),
            "surface": np.repeat(surface, 10),
            "end": np.repeat(end, 10),
            "depth": depth,
            "qc": qc,
            "fs": fs,
            "u2": u2,
        }
    )


@pytest.fixture
def cpt_data():
    """
    Fixture containing a DiscreteData instance of synthetic CPT data.

    """
    df = pd.concat([cpt_a(), cpt_b()], ignore_index=True)
    return DiscreteData(df)


@pytest.fixture
def cpt_collection(cpt_data):
    return cpt_data.to_collection()


@pytest.fixture
def xarray_dataset():
    x = np.arange(4) + 0.5
    y = x[::-1]
    z = np.arange(-2, 0, 0.5) + 0.25

    strat = [
        [[2, 2, 2, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 2, 1, 1]],
        [[2, 2, 1, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 1, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1]],
    ]
    lith = [
        [[2, 3, 2, 1], [2, 3, 1, 1], [2, 1, 1, 1], [3, 2, 1, 1]],
        [[2, 2, 1, 1], [2, 2, 1, 1], [2, 1, 1, 1], [2, 3, 2, 1]],
        [[2, 3, 2, 1], [2, 1, 1, 3], [2, 2, 1, 1], [2, 2, 2, 1]],
        [[2, 2, 2, 1], [2, 1, 1, 1], [2, 2, 1, 3], [3, 2, 2, 1]],
    ]
    ds = xr.Dataset(
        data_vars=dict(strat=(["y", "x", "z"], strat), lith=(["y", "x", "z"], lith)),
        coords=dict(y=y, x=x, z=z),
    )
    ds.rio.write_crs(28992, inplace=True)
    return ds


@pytest.fixture
def voxelmodel(xarray_dataset):
    return VoxelModel(xarray_dataset)


def create_polygons() -> gpd.GeoDataFrame:
    """
    Helper function to create a GeoDataFrame with 10 random irregular shaped polygons to
    create test fixtures with.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with only a "geometry" column containing the polygons.

    """
    points = [
        [0, 0],
        [0, 4],
        [4, 4],
        [4, 0],
        [0.5, 0.5],
        [0.7, 3],
        [1, 1.5],
        [2.1, 2.3],
        [2.7, 0.8],
        [3.1, 3.1],
        [3.3, 1.9],
        [1.8, 3.5],
        [1.5, 0.8],
        [1.4, 1.0],
    ]
    points = gpd.GeoDataFrame(geometry=[gmt.Point(p) for p in points], crs=28992)
    points["geometry"] = points.voronoi_polygons()
    points = points.clip((0, 0, 4, 4), sort=True)
    return points


@pytest.fixture
def simple_soilmap_gpkg(tmp_path):
    """
    Fixture to create a tmp geopackage file that contains relevant BRO soilmap information
    to test.

    """
    polygons = create_polygons()
    maparea_id = np.arange(len(polygons))
    soilunits = [
        "pVc",  # Peat type
        "hVk",  # Peat type
        "kVc",  # Peat type
        "Vc",  # Peat type
        "AAP",  # Peat type
        "vWp",  # Moerig type
        "iWp",  # Moerig type
        "kWz",  # Moerig type
        "AWv",  # Moerig type
        "Rv01C",  # Buried type
        "pRv81",  # Buried type
        "Mv51A",  # Buried type
        "Mv81A",  # Buried type
        "bEZ23",  # Other type
    ]
    polygons["maparea_id"] = maparea_id
    soilcodes = gpd.GeoDataFrame({"maparea_id": maparea_id, "soilunit_code": soilunits})

    layers = ["soilarea", "soilarea_soilunit"]
    tables = [polygons, soilcodes]

    outfile = tmp_path / "soilmap.gpkg"
    for layer, table in zip(layers, tables):
        table.to_file(outfile, driver="GPKG", layer=layer, index=False)

    return outfile


@pytest.fixture
def bro_cpt_gpkg(testdatadir):
    """
    Small extraction of 4 CPTs from the BRO CPT geopackage for testing purposes. The CPTs
    were selected from the original BRO CPT geopackage by their primary keys. The selected
    keys were: 164970, 164971, 164975, 164976. These numbers coincide with the "fid" index
    in the GeoDataFrame when `geost.bro.BroCptGeopackage` is used to read the geopackage.

    """
    return testdatadir / r"test_bro_cpt_geopackage.gpkg"
