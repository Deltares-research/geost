import pytest
from numpy.testing import assert_array_equal

from geost.bro.bro_geotop import Lithology


class TestUnitEnum:
    @pytest.mark.unittest
    def test_select_units(self):
        sel = Lithology.select_units(["clay", "fine_sand"])
        assert_array_equal(sel, [Lithology.clay, Lithology.fine_sand])
        assert_array_equal(sel, [2, 5])

        sel = Lithology.select_units("organic")
        assert_array_equal(sel, [Lithology.organic])
        assert_array_equal(sel, [1])

    @pytest.mark.unittest
    def test_select_values(self):
        sel = Lithology.select_values([2, 5])
        assert_array_equal(sel, [Lithology.clay, Lithology.fine_sand])
        assert_array_equal(sel, [2, 5])

        sel = Lithology.select_values(1)
        assert_array_equal(sel, [Lithology.organic])
        assert_array_equal(sel, [1])

    @pytest.mark.unittest
    def test_select_units_empty(self):
        sel = Lithology.select_units(["foo"])
        assert not sel

    @pytest.mark.unittest
    def test_select_values_empty(self):
        sel = Lithology.select_values(9999)
        assert not sel
