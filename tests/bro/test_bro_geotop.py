import pytest
from numpy.testing import assert_array_equal

from geost.bro import GeoTop
from geost.bro.bro_geotop import StratGeotop


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


class TestStratGeotop:
    @pytest.mark.unittest
    def test_select_units(self):
        sel = StratGeotop.select_units(["AAOP", "OEC", "BXDE", "ANAWA"])
        assert_array_equal(
            sel,
            [
                StratGeotop.holocene.OEC,
                StratGeotop.channel.ANAWA,
                StratGeotop.older.BXDE,
                StratGeotop.antropogenic.AAOP,
            ],
        )
        assert_array_equal(sel, [1070, 6010, 3040, 1000])

        sel = StratGeotop.select_units("AEC")
        assert_array_equal(sel, [StratGeotop.channel.AEC])
        assert_array_equal(sel, [6000])

    @pytest.mark.unittest
    def test_select_values(self):
        sel = StratGeotop.select_values([1070, 6010, 3040, 1000])
        assert_array_equal(
            sel,
            [
                StratGeotop.holocene.OEC,
                StratGeotop.channel.ANAWA,
                StratGeotop.older.BXDE,
                StratGeotop.antropogenic.AAOP,
            ],
        )
        assert_array_equal(sel, [1070, 6010, 3040, 1000])

        sel = StratGeotop.select_values(6000)
        assert_array_equal(sel, [StratGeotop.channel.AEC])
        assert_array_equal(sel, [6000])

    @pytest.mark.unittest
    def test_select_units_empty(self):
        sel = StratGeotop.select_units("foo")
        assert not sel

    @pytest.mark.unittest
    def test_select_values_empty(self):
        sel = StratGeotop.select_values(-9999)
        assert not sel
