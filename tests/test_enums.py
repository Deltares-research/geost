import pytest
from numpy.testing import assert_array_equal

from geost.bro import Lithology


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

    @pytest.mark.unittest
    def test_to_dict(self):
        d = Lithology.to_dict(key="unit")
        assert d == {
            "anthropogenic": 0,
            "organic": 1,
            "clay": 2,
            "loam": 3,
            "fine_sand": 5,
            "medium_sand": 6,
            "coarse_sand": 7,
            "gravel": 8,
            "shells": 9,
        }
        d = Lithology.to_dict(key="value")
        assert d == {
            0: "anthropogenic",
            1: "organic",
            2: "clay",
            3: "loam",
            5: "fine_sand",
            6: "medium_sand",
            7: "coarse_sand",
            8: "gravel",
            9: "shells",
        }
        with pytest.raises(ValueError):
            Lithology.to_dict(key="foo")  # Must raise ValueError with invalid input key
