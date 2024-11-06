import importlib

import pandas as pd
import pooch

from geost.read import read_borehole_table

REGISTRY = pooch.create(
    path=pooch.os_cache("geost"),
    base_url="https://github.com/Deltares-research/geost/raw/feature/docs/data/",
    version=None,
    version_dev="main",
    env="GEOST_DATA_DIR",
)
REGISTRY.load_registry(importlib.resources.files("geost.data") / "registry.txt")


def boreholes_usp(pandas=False):
    """
    Return a `BoreholeCollection` with a selection of DINOloket boreholes in the area of
    the Utrecht Science Park to use in GeoST tutorials.

    Parameters
    ----------
    pandas : bool, optional
        If True, read the boreholes into a `Pandas DataFrame`. The default is False, this
        returns a `BoreholeCollection`.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`

    """
    filename = REGISTRY.fetch("borehole_usp.parquet")
    if pandas:
        return pd.read_parquet(filename)
    else:
        return read_borehole_table(filename)
