import pyproj
import pytest
from numpy.testing import assert_almost_equal

from geost import projections


class TestProjections:
    @pytest.mark.unittest
    def test_get_horizontal_transformer(self):
        t = projections.horizontal_reference_transformer(28992, 4326)
        assert isinstance(t, pyproj.transformer.Transformer)

    @pytest.mark.unittest
    def test_get_vertical_transformer(self):
        t = projections.vertical_reference_transformer(28992, 5709, 5710)
        assert isinstance(t, pyproj.transformer.Transformer)

    @pytest.mark.unittest
    def test_xy_to_ll(self):
        lat_dec, lon_dec = projections.xy_to_ll(141000, 455000, 28992)
        assert_almost_equal(lon_dec, 5.182956740521864)
        assert_almost_equal(lat_dec, 52.083091710433095)
