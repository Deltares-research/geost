import pytest
from pathlib import Path
from pysst import projections
from numpy.testing import assert_almost_equal


class TestProjections:
    def test_get_projection(self):
        P = projections.get_projection("28992")
        assert (
            P.definition
            == "proj=sterea lat_0=52.1561605555556 lon_0=5.38763888888889 k=0.9999079 x_0=155000 y_0=463000 ellps=bessel units=m no_defs"
        )

    def test_xy_to_ll(self):
        lon_dec, lat_dec = projections.xy_to_ll(141000, 455000, "28992")
        assert_almost_equal(lon_dec, 5.1833624)
        assert_almost_equal(lat_dec, 52.0840712)
