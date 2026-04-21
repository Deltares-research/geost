import importlib
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pooch
import xarray as xr

from geost.bro import GeoTop
from geost.io.read import read_borehole_table, read_cpt_table

REGISTRY = pooch.create(
    path=pooch.os_cache("geost"),
    base_url="https://github.com/Deltares-research/geost/raw/main/data/",
    version=None,
    env="GEOST_DATA_DIR",
)
REGISTRY.load_registry(importlib.resources.files("geost.data") / "registry.txt")


def boreholes_usp(pandas=False, return_filepath=False):
    """
    Return a :class:`~geost.base.Collection` with a selection of DINOloket boreholes
    in the area of the Utrecht Science Park to use in GeoST tutorials.

    Parameters
    ----------
    pandas : bool, optional
        If True, read the boreholes into a `Pandas DataFrame`. The default is False, this
        returns a :class:`~geost.base.Collection`.
    return_filepath : bool, optional
        If True, return the file path to the borehole data instead of reading it. The
        default is False.

    Returns
    -------
    :class:`~geost.base.Collection`

    """
    filename = REGISTRY.fetch("boreholes_usp.parquet")
    if return_filepath:
        return Path(filename)

    if pandas:
        return pd.read_parquet(filename)
    else:
        return read_borehole_table(
            filename, coll_kwargs={"crs": 28992, "vertical_reference": 5709}
        )


def cpts_usp(pandas=False, return_filepath=False):
    """
    Return a :class:`~geost.base.Collection` with a selection of BROloket CPTs in the
    area of the Utrecht Science Park to use in GeoST tutorials.

    Parameters
    ----------
    pandas : bool, optional
        If True, read the CPTs into a `Pandas DataFrame`. The default is False, this
        returns a :class:`~geost.base.Collection`.
    return_filepath : bool, optional
        If True, return the file path to the CPT data instead of reading it. The default
        is False.

    Returns
    -------
    :class:`~geost.base.Collection`

    """
    filename = REGISTRY.fetch("cpts_usp.parquet")
    if return_filepath:
        return Path(filename)

    if pandas:
        return pd.read_parquet(filename)
    else:
        return read_cpt_table(
            filename, coll_kwargs={"crs": 28992, "vertical_reference": 5709}
        )


def geotop_usp(xarray=False, return_filepath=False):
    """
    Return a :class:`~geost.bro.GeoTop` instance in the area of the Utrecht Science Park
    to use in GeoST tutorials.

    Parameters
    ----------
    xarray : bool, optional
        If True, read the GeoTOP data as an `xarray.Dataset`. The default is False, this
        returns a :class:`~geost.bro.GeoTop`.
    return_filepath : bool, optional
        If True, return the file path to the GeoTOP data instead of reading it. The
        default is False.

    Returns
    -------
    :class:`~geost.bro.GeoTop`

    """
    filename = REGISTRY.fetch("geotop_usp.nc")
    if return_filepath:
        return Path(filename)

    if xarray:
        return xr.open_dataset(filename)
    else:
        return GeoTop.from_netcdf(filename)


def bhrg_bro():
    """
    Download a BRO BHR-G XML file from the GeoST data registry and return a `Pathlib.Path`
    object pointing to the file for tutorials explaining GeoST's XML parsing.

    Returns
    -------
    Pathlib.Path
        A Pathlib.Path object pointing to the downloaded XML file.

    """
    return Path(REGISTRY.fetch("bhrg_bro.xml"))


def dike_section():
    """
    Download a dike section geoparquet file from the GeoST data registry and return a
    Geopandas `GeoDataFrame` of the dike section for tutorials.

    Returns
    -------
    GeoDataFrame
        A Geopandas GeoDataFrame object containing the dike section data.

    """
    filename = REGISTRY.fetch("dike_section.geoparquet")
    return gpd.read_parquet(filename)
