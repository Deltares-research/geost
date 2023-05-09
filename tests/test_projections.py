import pyproj
import pytest
from numpy.testing import assert_almost_equal

from pysst import projections


class TestProjections:
    @pytest.mark.unittest
    def test_get_transformer(self):
        T = projections.get_transformer(28992, 4326)
        assert type(T) == pyproj.transformer.Transformer

    @pytest.mark.unittest
    def test_xy_to_ll(self):
        lon_dec, lat_dec = projections.xy_to_ll(141000, 455000, 28992)
        assert_almost_equal(lon_dec, 5.182956740521864)
        assert_almost_equal(lat_dec, 52.083091710433095)
