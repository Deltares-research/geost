import pytest
import numpy as np
import pandas as pd
from numpy.testing import (
    assert_equal,
    assert_allclose,
    assert_array_equal,
    assert_almost_equal,
)

# Local imports
from pysst.analysis.interpret_cpt import calc_ic, calc_lithology
from pysst.analysis.layer_analysis import find_top_sand
from pysst.borehole import BoreholeCollection


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

    @pytest.fixture
    def borehole_collection(self):
        nr = np.full(10, "B-01")
        x = np.full(10, 139370)
        y = np.full(10, 455540)
        mv = np.full(10, 1.0)
        end = np.full(10, -4.0)
        top = np.array([1, 0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -3.6])
        bottom = np.array([0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -3.6, -4.0])
        data_string = np.array(
            ["K", "Kz", "K", "Ks3", "Ks2", "V", "Zk", "Zs", "Z", "Z"]
        )
        data_int = np.arange(0, 10, dtype=np.int64)
        data_float = np.arange(0, 5, 0.5, dtype=np.float64)

        dataframe = pd.DataFrame(
            {
                "nr": nr,
                "x": x,
                "y": y,
                "mv": mv,
                "end": end,
                "top": top,
                "bottom": bottom,
                "data_string": data_string,
                "data_int": data_int,
                "data_float": data_float,
            }
        )
        return BoreholeCollection(dataframe)

    @pytest.mark.unittest
    def test_top_sand(
        self, test_borehole_lith, test_borehole_top, test_borehole_bottom
    ):
        # After the first encounter of sand (Z) > 40% of the next 1 m must consist of sand as well
        # for this first encounter to be regarded as the top of the sand (and all the above the cover layer)
        assert_equal(
            find_top_sand(
                test_borehole_lith, test_borehole_top, test_borehole_bottom, 0.4, 1
            ),
            -3.0,
        )

        # After the first encounter of sand (Z) > 60% of the next 1 m must consist of sand as well
        # for this first encounter to be regarded as the top of the sand (and all the above the cover layer)
        assert_equal(
            find_top_sand(
                test_borehole_lith, test_borehole_top, test_borehole_bottom, 0.6, 1
            ),
            -4.0,
        )

    @pytest.mark.unittest
    def test_cumulative_thickness(self, borehole_collection):
        assert_almost_equal(
            borehole_collection.get_cumulative_layer_thickness("data_string", "K")[
                "K_thickness"
            ][0],
            1.0,
        )
        assert_almost_equal(
            borehole_collection.get_cumulative_layer_thickness("data_string", "Z")[
                "Z_thickness"
            ][0],
            0.8,
        )
        assert_almost_equal(
            borehole_collection.get_cumulative_layer_thickness("data_string", "V")[
                "V_thickness"
            ][0],
            0.5,
        )

    @pytest.mark.unittest
    def test_layer_top(self, borehole_collection):
        assert_almost_equal(
            borehole_collection.get_layer_top("data_string", "K")["K_top"][0],
            1.0,
        )
        assert_almost_equal(
            borehole_collection.get_layer_top("data_string", "Z")["Z_top"][0],
            -3.2,
        )
        assert_almost_equal(
            borehole_collection.get_layer_top("data_string", "V")["V_top"][0],
            -2.0,
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
