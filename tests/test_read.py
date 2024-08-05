from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost import (
    get_bro_objects_from_bbox,
    get_bro_objects_from_geometry,
    read_nlog_cores,
    read_sst_cores,
)
from geost.borehole import BoreholeCollection


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
                "surface": [36.7, 39.54, 33.51],
                "end": [-3921.75, -3262.69, -3865.89],
            }
        )
        assert_array_equal(
            nlog_cores.header[["nr", "x", "y", "surface", "end"]], desired_df
        )
        assert nlog_cores.has_inclined

    @pytest.mark.unittest
    def test_nlog_reader_from_parquet(self):
        nlog_cores = read_nlog_cores(self.nlog_stratstelsel_parquet)
        desired_df = pd.DataFrame(
            {
                "nr": ["BA050018", "BA080036", "BA110040"],
                "x": [37395, 44175, 39309],
                "y": [857077, 840614, 833198],
                "surface": [36.7, 39.54, 33.51],
                "end": [-3921.75, -3262.69, -3865.89],
            }
        )
        assert_array_equal(
            nlog_cores.header[["nr", "x", "y", "surface", "end"]], desired_df
        )
        assert nlog_cores.data.has_inclined

    @pytest.mark.unittest
    def test_get_bro_soil_cores_from_bbox(self):
        soilcores = get_bro_objects_from_bbox(
            "BHR-P", xmin=87000, xmax=87500, ymin=444000, ymax=444500
        )
        assert soilcores.n_points == 7

    @pytest.mark.unittest
    def test_get_bro_soil_cores_from_geometry(self):
        soilcores = get_bro_objects_from_geometry(
            "BHR-P",
            Path(__file__).parent / "data/test_polygon.parquet",
        )
        assert soilcores.n_points == 1

    @pytest.mark.unittest
    def test_read_dino(self):
        dino = read_sst_cores(
            r"c:\Users\onselen\Lokale data\DINO_Extractie_bovennaaronder_d20230201.parquet",
            column_mapper={"surface": "mv"},
        )
        print("stop")
