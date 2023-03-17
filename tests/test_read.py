import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
import pandas as pd

from pysst.readers import pygef_gef_cpt


# class TestReaders:
#     @pytest.mark.unittest
#     def test_read_pygef_gef_cpt(self):
#         test_file_path = Path(
#             Path(__file__).parent / "data/CPT000000038871_IMBRO_A.gef"
#         )
#         df = pd.concat(pygef_gef_cpt(test_file_path))
#         # Checks
#         assert_equal(df["nr"][0], "CPT000000038871")
#         assert_equal(df["x"][0], 127160.0)
#         assert_equal(df["y"][0], 502173.0)
#         assert_equal(df["mv"][0], -2.7)
#         assert_almost_equal(df["end"][0], -22.619999999999997)
#         assert_allclose(df["top"].head(), np.array([-4.38, -4.40, -4.42, -4.44, -4.46]))
#         assert_allclose(
#             df["top"].tail(), np.array([-22.52, -22.54, -22.56, -22.58, -22.60])
#         )
#         assert_allclose(
#             df["bottom"].head(), np.array([-4.40, -4.42, -4.44, -4.46, -4.48])
#         )
#         assert_allclose(
#             df["bottom"].tail(), np.array([-22.54, -22.56, -22.58, -22.60, -22.62])
#         )
#         assert_allclose(df["qc"].head(), np.array([0.196, 0.183, 0.170, 0.163, 0.147]))
#         assert_allclose(
#             df["friction_number"].head(),
#             np.array([5.102041, 6.010929, 6.470588, 7.361963, 7.482993]),
#         )
