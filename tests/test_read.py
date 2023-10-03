from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from pysst import read_nlog_cores
from pysst.borehole import BoreholeCollection


class TestReaders:
    export_folder = Path(__file__).parent / "data"
    nlog_stratstelsel_xlsx = (
        Path(__file__).parent / "data/test_nlog_stratstelsel_20230807.xlsx"
    )
    nlog_stratstelsel_parquet = (
        Path(__file__).parent / "data/test_nlog_stratstelsel_20230807.parquet"
    )

    @pytest.mark.unittest
    def test_nlog_reader_from_excel(self):
        nlog_cores = read_nlog_cores(self.nlog_stratstelsel_xlsx)
        desired_df = pd.DataFrame(
            {
                "nr": ["BA050018", "BA080036", "BA110040"],
                "x": [37395, 44175, 39309],
                "y": [857077, 840614, 833198],
                "mv": [36.7, 39.54, 33.51],
                "end": [-3921.75, -3262.69, -3865.89],
            }
        )
        assert_array_equal(nlog_cores.header[["nr", "x", "y", "mv", "end"]], desired_df)
        assert nlog_cores.is_inclined

    @pytest.mark.unittest
    def test_nlog_reader_from_parquet(self):
        nlog_cores = read_nlog_cores(self.nlog_stratstelsel_parquet)
        desired_df = pd.DataFrame(
            {
                "nr": ["BA050018", "BA080036", "BA110040"],
                "x": [37395, 44175, 39309],
                "y": [857077, 840614, 833198],
                "mv": [36.7, 39.54, 33.51],
                "end": [-3921.75, -3262.69, -3865.89],
            }
        )
        assert_array_equal(nlog_cores.header[["nr", "x", "y", "mv", "end"]], desired_df)
        assert nlog_cores.is_inclined
