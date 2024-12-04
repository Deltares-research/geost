import importlib

import pandas as pd
import pooch
import xarray as xr

from geost.bro import GeoTop
from geost.read import read_borehole_table, read_cpt_table

REGISTRY = pooch.create(
    path=pooch.os_cache("geost"),
    base_url="https://github.com/Deltares-research/geost/raw/feature/docs/data/",  # TODO: Change to 'main' branch
    version=None,
    env="GEOST_DATA_DIR",
)
REGISTRY.load_registry(importlib.resources.files("geost.data") / "registry.txt")


def boreholes_usp(pandas=False):
    """
    Return a :class:`~geost.base.BoreholeCollection` with a selection of DINOloket boreholes
    in the area of the Utrecht Science Park to use in GeoST tutorials.

    Parameters
    ----------
    pandas : bool, optional
        If True, read the boreholes into a `Pandas DataFrame`. The default is False, this
        returns a :class:`~geost.base.BoreholeCollection`.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`

    """
    filename = REGISTRY.fetch("boreholes_usp.parquet")
    if pandas:
        return pd.read_parquet(filename)
    else:
        return read_borehole_table(filename)


def cpts_usp(pandas=False):
    """
    Return a :class:`~geost.base.CptCollection` with a selection of BROloket CPTs in the
    area of the Utrecht Science Park to use in GeoST tutorials.

    Parameters
    ----------
    pandas : bool, optional
        If True, read the boreholes into a `Pandas DataFrame`. The default is False, this
        returns a :class:`~geost.base.CptCollection`.

    Returns
    -------
    :class:`~geost.base.CptCollection`

    """
    filename = REGISTRY.fetch("cpts_usp.parquet")
    if pandas:
        return pd.read_parquet(filename)
    else:
        return read_cpt_table(filename)


def geotop_usp(xarray=False):
    """
    Return a :class:`~geost.bro.GeoTop` instance in the area of the Utrecht Science Park
    to use in GeoST tutorials.

    Parameters
    ----------
    xarray : bool, optional
        If True, read the GeoTOP data as an `xarray.Dataset`. The default is False, this
        returns a :class:`~geost.bro.GeoTop`.

    Returns
    -------
    :class:`~geost.bro.GeoTop`

    """
    filename = REGISTRY.fetch("geotop_usp.nc")
    if xarray:
        return xr.open_dataset(filename)
    else:
        return GeoTop.from_netcdf(filename)
