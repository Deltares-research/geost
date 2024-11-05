import numpy as np
import pandas as pd
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

# Local imports
from geost.analysis.combine import add_voxelmodel_variable
from geost.analysis.interpret_cpt import calc_ic, calc_lithology
from geost.analysis.layer_analysis import find_top_sand
from geost.base import Collection


class TestAnalysis:
    @pytest.fixture
    def test_borehole_lith(self):
        return np.array(["K", "V", "K", "Z", "K", "Z", "Z"])

    @pytest.fixture
    def test_borehole_top(self):
        return np.array([0, -1, -1.5, -3, -3.5, -4, -6])

    @pytest.fixture
    def test_borehole_bottom(self):
        return np.array([-1, -1.5, -3, -3.5, -4, -6, -10])

    @pytest.fixture
    def test_ic_array(self):
        return np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

    @pytest.fixture
    def test_fr_array(self):
        return np.array([9.0, 7.0, 5.5, 5.0, 4.0, 3.0, 3.0, 2.0, 1.5, 1.0, 1.0])

    @pytest.mark.unittest
    def test_top_sand(
        self, test_borehole_lith, test_borehole_top, test_borehole_bottom
    ):
        # After the first encounter of sand (Z) > 40% of the next 1 m must consist of
        # sand as well for this first encounter to be regarded as the top of the sand
        # (and all the above the cover layer)
        assert_equal(
            find_top_sand(
                test_borehole_lith, test_borehole_top, test_borehole_bottom, 0.4, 1
            ),
            -3.0,
        )

        # After the first encounter of sand (Z) > 60% of the next 1 m must consist of
        # sand as well for this first encounter to be regarded as the top of the sand
        # (and all the above the cover layer)
        assert_equal(
            find_top_sand(
                test_borehole_lith, test_borehole_top, test_borehole_bottom, 0.6, 1
            ),
            -4.0,
        )

    @pytest.mark.unittest
    def test_calc_ic(self, test_ic_array, test_fr_array):
        ic = calc_ic(test_ic_array, test_fr_array)
        assert_allclose(
            ic,
            np.array(
                [
                    4.09490299,
                    4.03801064,
                    3.72631088,
                    3.55524529,
                    3.39779083,
                    3.249435,
                    2.99685177,
                    2.64914385,
                    2.25513149,
                    1.91031411,
                    1.68964223,
                ]
            ),
        )

    @pytest.mark.unittest
    def test_calc_lithology(self, test_ic_array, test_fr_array):
        lith = calc_lithology(
            calc_ic(test_ic_array, test_fr_array), test_ic_array, test_fr_array
        )
        assert_array_equal(
            lith, np.array(["V", "Kh", "Kh", "K", "K", "K", "K", "Kz", "Z", "Z", "Z"])
        )


class TestCombine:
    @pytest.mark.unittest
    def test_add_voxelmodel_variable_layered(self, borehole_collection, voxelmodel):
        result = add_voxelmodel_variable(borehole_collection, voxelmodel, "strat")
        assert isinstance(result, Collection)
        assert_array_equal(
            result.data["strat"],
            [
                1,
                1,
                1,
                2,
                np.nan,
                np.nan,
                np.nan,
                1,
                1,
                1,
                2,
                np.nan,
                np.nan,
                np.nan,
                1,
                2,
                2,
                2,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1,
                1,
                1,
                2,
                2,
                np.nan,
                np.nan,
            ],
        )
        assert_array_almost_equal(
            result.data["top"],
            [
                0.0,
                0.8,
                1.5,
                1.7,
                2.2,
                2.5,
                3.7,
                0.0,
                0.6,
                1.2,
                1.3,
                2.3,
                2.5,
                3.1,
                0.0,
                0.75,
                1.4,
                1.8,
                2.25,
                2.9,
                3.8,
                0.0,
                0.5,
                1.2,
                1.8,
                2.5,
                0.0,
                0.5,
                1.2,
                1.4,
                1.8,
                1.9,
                2.5,
            ],
        )
        assert_array_almost_equal(
            result.data["bottom"],
            [
                0.8,
                1.5,
                1.7,
                2.2,
                2.5,
                3.7,
                4.2,
                0.6,
                1.2,
                1.3,
                2.3,
                2.5,
                3.1,
                3.9,
                0.75,
                1.4,
                1.8,
                2.25,
                2.9,
                3.8,
                5.5,
                0.5,
                1.2,
                1.8,
                2.5,
                3.0,
                0.5,
                1.2,
                1.4,
                1.8,
                1.9,
                2.5,
                3.0,
            ],
        )

    @pytest.mark.unittest
    def test_add_voxelmodel_variable_discrete(self, cpt_collection, voxelmodel):
        result = add_voxelmodel_variable(cpt_collection, voxelmodel, "strat")
        assert isinstance(result, Collection)
        assert_array_equal(
            result.data["strat"],
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        )
        assert_array_equal(
            result.data["depth"],
            [
                0.0,
                1.0,
                2.0,
                3.0,
                3.6,
                4.0,
                4.1,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                0.0,
                1.0,
                2.0,
                2.3,
                2.8,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
        )
