from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from geost.io.parsers import CptGefFile
from geost.io.parsers.gef_parsers import ColumnInfo


def dummy_cpt_data():
    z = -0.5
    columns = ["length", "qc", "fs", "rf"]
    columninfo = {ii: ColumnInfo(c, None, None, None) for ii, c in enumerate(columns)}
    columnvoid = {0: -9999.0, 1: -9999.0, 2: -999.0, 3: -9999.0}

    length = ["1.300000", "1.320000", "1.340000", "1.360000", "1.380000"]
    qc = ["-9999.0000", "0.227523", "0.279521", "0.327816", "-9999.0000"]
    fs = ["-999.000", "0.010076", "0.014140", "-999.000", "0.020937"]
    rf = ["-9999.0000", "3.561200", "4.462611", "5.452729", "5.697089"]

    data = np.array([length, qc, fs, rf]).T

    return z, columninfo, columnvoid, data


class CptInfo(NamedTuple):
    nr: str
    x: float
    y: float
    z: float
    end: float
    ncols: int
    system: str
    reference: str
    error: float
    nrecords: int


class TestCptGefParser:
    @pytest.fixture
    def test_cpt_files(self, testdatadir):
        test_cpt_files = list(testdatadir.glob("gef/*.gef"))
        return test_cpt_files

    @pytest.fixture
    def cpt_a(self, testdatadir):
        cpt = CptGefFile(testdatadir / r"gef/83268_DKMP003_wiertsema.gef")
        info_to_test = CptInfo(
            "DKMP_D03",
            176161.1,
            557162.1,
            -0.06,
            -0.06 - 37.317295,
            11,
            "31000",
            "NAP",
            0.01,
            1815,
        )
        return cpt, info_to_test

    @pytest.fixture
    def cpt_b(self, testdatadir):
        cpt = CptGefFile(testdatadir / r"gef/AZZ158_gem_rotterdam.gef")
        info_to_test = CptInfo(
            "AZZ158",
            0.0,
            0.0,
            5.05,
            5.05 - 59.5,
            6,
            "0",
            "NAP",
            None,
            2980,
        )
        return cpt, info_to_test

    @pytest.fixture
    def cpt_c(self, testdatadir):
        cpt = CptGefFile(testdatadir / r"gef/CPT000000157983_IMBRO.gef")
        info_to_test = CptInfo(
            "CPT000000157983",
            176416.1,
            557021.9,
            -5.5,
            -5.5 - 28.84,
            9,
            "28992",
            "NAP",
            None,
            1310,
        )
        return cpt, info_to_test

    @pytest.fixture
    def cpt_d(self, testdatadir):
        cpt = CptGefFile(testdatadir / r"gef/CPT10.gef")
        info_to_test = CptInfo(
            "YANGTZEHAVEN CPT 10",
            61949.0,
            443624.0,
            -17.69,
            -17.69 - 5.75,
            4,
            "31000",
            "NAP",
            None,
            207,
        )
        return cpt, info_to_test

    @pytest.fixture
    def dummy_cpt_with_rf(self):
        z, columninfo, columnvoid, data = dummy_cpt_data()

        cpt = CptGefFile()
        cpt.z = z
        cpt.columninfo = columninfo
        cpt.columnvoid = columnvoid
        cpt._data = data

        return cpt

    @pytest.mark.unittest
    def test_read_files(self, test_cpt_files: list[Path]):
        for f in test_cpt_files:
            cpt = CptGefFile(f)
            assert isinstance(cpt, CptGefFile)

    @pytest.mark.integrationtest
    @pytest.mark.parametrize("cpt_test", ["cpt_a", "cpt_b", "cpt_c", "cpt_d"])
    def test_cpt_parsing_result(
        self,
        cpt_test: (
            Literal["cpt_a"] | Literal["cpt_b"] | Literal["cpt_c"] | Literal["cpt_d"]
        ),
        request: pytest.FixtureRequest,
    ):
        cpt, test_info = request.getfixturevalue(cpt_test)

        assert cpt.nr == test_info.nr
        assert cpt.x == test_info.x
        assert cpt.y == test_info.y
        assert cpt.z == test_info.z
        assert cpt.enddepth == test_info.end
        assert cpt.ncolumns == test_info.ncols
        assert cpt.coord_system == test_info.system
        assert cpt.reference_system == test_info.reference
        assert cpt.delta_z == test_info.error

        critical_cols_from_file = ["length", "qc", "fs"]
        assert all(col in cpt.columns for col in critical_cols_from_file)
        assert len(cpt.df) == test_info.nrecords

    @pytest.mark.unittest
    def test_to_dataframe(self, dummy_cpt_with_rf: CptGefFile):
        dummy_cpt_with_rf.to_df()

        target_columns = ["length", "qc", "fs", "rf", "depth"]
        target_depth = [1.30, 1.32, 1.34, 1.36, 1.38]

        df = dummy_cpt_with_rf._df
        nancount = np.sum(np.isnan(df.values))

        assert all(df.dtypes == "float64")
        assert nancount == 5
        assert all(df.columns == target_columns)

        assert_array_almost_equal(df["depth"], target_depth)
