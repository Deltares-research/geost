from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

import geost


@pytest.mark.unittest
def test_boreholes_usp():
    boreholes = geost.data.boreholes_usp()
    assert isinstance(boreholes, geost.BoreholeCollection)
    boreholes = geost.data.boreholes_usp(pandas=True)
    assert isinstance(boreholes, pd.DataFrame)


@pytest.mark.unittest
def test_cpts_usp():
    cpts = geost.data.cpts_usp()
    assert isinstance(cpts, geost.CptCollection)
    cpts = geost.data.cpts_usp(pandas=True)
    assert isinstance(cpts, pd.DataFrame)


@pytest.mark.unittest
def test_geotop_usp():
    gtp = geost.data.geotop_usp()
    assert isinstance(gtp, geost.bro.GeoTop)
    gtp = geost.data.geotop_usp(xarray=True)
    assert isinstance(gtp, xr.Dataset)


@pytest.mark.unittest
def test_bhrg_bro():
    bhrg = geost.data.bhrg_bro()
    assert isinstance(bhrg, Path)
    assert bhrg.is_file()


@pytest.mark.unittest
def test_dike_section():
    dike = geost.data.dike_section()
    assert isinstance(dike, gpd.GeoDataFrame)
