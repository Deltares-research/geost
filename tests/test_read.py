from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost import (
    get_bro_objects_from_bbox,
    get_bro_objects_from_geometry,
    read_borehole_table,
    read_nlog_cores,
)
from geost.base import BoreholeCollection, LayeredData
from geost.read import (
    MANDATORY_LAYERED_DATA_COLUMNS,
    _check_mandatory_column_presence,
    adjust_z_coordinates,
)


class TestReaders:
    export_folder = Path(__file__).parent / "data"
    nlog_stratstelsel_xlsx = (
        Path(__file__).parent / "data/test_nlog_stratstelsel_20230807.xlsx"
    )
    nlog_stratstelsel_parquet = (
        Path(__file__).parent / "data/test_nlog_stratstelsel_20230807.parquet"
    )

    @pytest.fixture
    def table_wrong_columns(self):
        return pd.DataFrame(
            {
                "nr": ["a", "b"],
                "x": [1, 2],
                "y": [1, 2],
                "maaiveld": [1, 2],
                "end": [1, 1],
                "top": [1, 1],
                "bottom": [2, 2],
            }
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
    def test_read_borehole_table(self):
        file_pq = Path(__file__).parent / r"data/test_borehole_table.parquet"
        file_csv = Path(__file__).parent / r"data/test_borehole_table.csv"
        cores_pq = read_borehole_table(file_pq)
        cores_csv = read_borehole_table(file_csv)
        assert isinstance(cores_pq, BoreholeCollection)
        assert isinstance(cores_csv, BoreholeCollection)

        cores_pq = read_borehole_table(file_pq, as_collection=False)
        cores_csv = read_borehole_table(file_csv, as_collection=False)
        assert isinstance(cores_pq, LayeredData)
        assert isinstance(cores_csv, LayeredData)

    @pytest.mark.unittest
    def test_read_inclined_borehole_table(self):
        file_pq = Path(__file__).parent / r"data/test_inclined_borehole_table.parquet"
        cores_pq = read_borehole_table(file_pq, has_inclined=True)
        assert isinstance(cores_pq, BoreholeCollection)

        cores_pq = read_borehole_table(file_pq, has_inclined=True, as_collection=False)
        assert isinstance(cores_pq, LayeredData)

    @pytest.mark.unittest
    def test_check_mandatory_columns(self, table_wrong_columns):
        column_mapper = {"maaiveld": "surface"}
        table_wrong_columns = _check_mandatory_column_presence(
            table_wrong_columns, MANDATORY_LAYERED_DATA_COLUMNS, column_mapper
        )
        assert_array_equal(table_wrong_columns.columns, MANDATORY_LAYERED_DATA_COLUMNS)

    @pytest.mark.unittest
    def test_check_mandatory_columns_with_user_input(
        self, table_wrong_columns, monkeypatch
    ):
        monkeypatch.setattr("builtins.input", lambda _: "maaiveld")
        table_wrong_columns = _check_mandatory_column_presence(
            table_wrong_columns, MANDATORY_LAYERED_DATA_COLUMNS
        )
        assert_array_equal(table_wrong_columns.columns, MANDATORY_LAYERED_DATA_COLUMNS)

    @pytest.mark.unittest
    def test_adjust_z_coordinates(self):
        file = Path(__file__).parent / r"data/test_borehole_table.parquet"
        cores = read_borehole_table(file, as_collection=False)
        cores_df = cores.df.copy()

        # test situation where top and bottoms are already positive downward
        cores_df_adjusted = adjust_z_coordinates(cores_df.copy())
        assert cores_df_adjusted["top"].iloc[0] < cores_df_adjusted["top"].iloc[1]
        assert cores_df_adjusted["bottom"].iloc[0] < cores_df_adjusted["bottom"].iloc[1]

        # Test situation where top and bottoms are given as negative downward
        cores_df["top"] *= -1
        cores_df["bottom"] *= -1

        cores_df_adjusted = adjust_z_coordinates(cores_df.copy())
        assert cores_df_adjusted["top"].iloc[0] < cores_df_adjusted["top"].iloc[1]
        assert cores_df_adjusted["bottom"].iloc[0] < cores_df_adjusted["bottom"].iloc[1]
