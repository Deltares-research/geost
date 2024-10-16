import pytest
from numpy.testing import assert_array_equal

from geost.bro import GeoTop


class TestGeoTop:
    @pytest.fixture
    def test_area(self):
        """
        xmin, ymin, xmax, ymax bounding box of a test area.
        """
        return (115_000, 500_000, 115_500, 500_500)

    @pytest.mark.unittest
    def test_from_opendap(self, test_area):
        geotop = GeoTop.from_opendap(bbox=test_area)
        assert isinstance(geotop, GeoTop)
        assert geotop.resolution == (100, 100, 0.5)
        assert geotop["strat"].dims == ("y", "x", "z")
        assert geotop.crs == 28992
        assert_array_equal(geotop["x"], [115_050, 115_150, 115_250, 115_350, 115_450])
        assert_array_equal(geotop["y"], [500_450, 500_350, 500_250, 500_150, 500_050])
        assert_array_equal(geotop["z"][:3], [-49.75, -49.25, -48.75])
