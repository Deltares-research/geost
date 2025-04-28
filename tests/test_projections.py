import pyproj
import pytest
from numpy.testing import assert_almost_equal

from geost import projections


@pytest.mark.unittest
def test_get_horizontal_transformer():
    t = projections.horizontal_reference_transformer(28992, 4326)
    assert isinstance(t, pyproj.transformer.Transformer)


@pytest.mark.unittest
def test_get_vertical_transformer():
    t = projections.vertical_reference_transformer(28992, 5709, 5710)
    assert isinstance(t, pyproj.transformer.Transformer)


@pytest.mark.unittest
def test_projected_to_geographic():
    t = projections.horizontal_reference_transformer(28992, 4326)
    x, y = 140_000, 440_000
    lon, lat = t.transform(x, y)
    assert_almost_equal(lon, 5.169024734261481)
    assert_almost_equal(lat, 51.948244431668776)


@pytest.mark.unittest
def test_geographic_to_projected():
    t = projections.horizontal_reference_transformer(4326, 28992)
    lon, lat = 5.1, 51.9
    x, y = t.transform(lon, lat)
    assert_almost_equal(x, 135233.28064173012)
    assert_almost_equal(y, 434649.01993844495)
