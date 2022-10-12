import pytest
import numpy as np

# Local imports
from pysst.analysis.interpret_cpt import calc_ic, calc_lithology


class TestInterpretCpt:
    @pytest.fixture
    def test_ic_array(self):
        return np.arange(0, 20, 0.1)

    @pytest.fixture
    def test_fr_array(self):
        return np.arange(10, 0, -0.05)

    @pytest.mark.unittest
    def test_calc_ic(self, test_ic_array, test_fr_array):
        ic = calc_ic(test_ic_array, test_fr_array)
