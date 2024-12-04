import pandas as pd
import pytest
import xarray as xr

import geost
from geost.base import BoreholeCollection, CptCollection
from geost.bro import GeoTop


@pytest.mark.unittest
def test_boreholes_usp():
    boreholes = geost.data.boreholes_usp()
    assert isinstance(boreholes, BoreholeCollection)
    boreholes = geost.data.boreholes_usp(pandas=True)
    assert isinstance(boreholes, pd.DataFrame)


@pytest.mark.unittest
def test_cpts_usp():
    cpts = geost.data.cpts_usp()
    assert isinstance(cpts, CptCollection)
    cpts = geost.data.cpts_usp(pandas=True)
    assert isinstance(cpts, pd.DataFrame)


@pytest.mark.unittest
def test_geotop_usp():
    gtp = geost.data.geotop_usp()
    assert isinstance(gtp, GeoTop)
    gtp = geost.data.geotop_usp(xarray=True)
    assert isinstance(gtp, xr.Dataset)
