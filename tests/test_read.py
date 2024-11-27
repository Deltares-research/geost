from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost import (
    get_bro_objects_from_bbox,
    get_bro_objects_from_geometry,
    read_borehole_table,
    read_collection_geopackage,
    read_cpt_table,
    read_gef_cpts,
    read_nlog_cores,
    read_pickle,
    read_uullg_tables,
    read_xml_boris,
)
from geost.base import (
    BoreholeCollection,
    CptCollection,
    DiscreteData,
    LayeredData,
    PointHeader,
)
from geost.read import (
    MANDATORY_LAYERED_DATA_COLUMNS,
    _check_mandatory_column_presence,
    adjust_z_coordinates,
)


class InvalidCollection:
    """
    Invalid Collection type to raise a ValueError in `read_collection_geopackage`.
    """

    pass


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def llg_header_table(tmp_path):
    outfile = tmp_path / r"llg_header.parquet"
    header = pd.DataFrame(
        {
            "BOORP": [1901, 1902],
            "XCO": [1, 2],
            "YCO": [1, 2],
            "MV_HoogteNAP": [2.1, 1.9],
            "BoringEinddiepteNAP": [1.7, 1.5],
        }
    )
    header.to_parquet(outfile)
    return outfile


@pytest.fixture
def llg_data_table(tmp_path):
    outfile = tmp_path / r"llg_data.parquet"
    data = pd.DataFrame(
        {
            "BOORP": [1901, 1901, 1902, 1902],
            "BEGIN_DIEPTE": [0, 30, 0, 20],
            "EIND_DIEPTE": [30, 40, 20, 40],
            "TEXTUUR": ["ZK", "L", "Z", "K"],
        }
    )
    data.to_parquet(outfile)
    return outfile


@pytest.fixture
def llg_data_table_duplicate_column(tmp_path):
    outfile = tmp_path / r"llg_data_dupl.parquet"
    data = pd.DataFrame(
        {
            "nr": [1901, 1901, 1902, 1902],
            "top": [0, 30, 0, 20],
            "bottom": [30, 40, 20, 40],
            "TEXTUUR": ["ZK", "L", "Z", "K"],
        }
    )
    data.to_parquet(outfile)
    return outfile


@pytest.fixture
def collection_pickle(borehole_collection, tmp_path):
    outfile = tmp_path / r"collection.pkl"
    borehole_collection.to_pickle(outfile)
    return outfile


class TestReaders:
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
    def test_nlog_reader_from_parquet(self, data_dir):
        nlog_cores = read_nlog_cores(
            data_dir / r"test_nlog_stratstelsel_20230807.parquet"
        )
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
    def test_get_bro_soil_cores_from_geometry(self, data_dir):
        soilcores = get_bro_objects_from_geometry(
            "BHR-P", data_dir / "test_polygon.parquet"
        )
        assert soilcores.n_points == 1

    @pytest.mark.unittest
    def test_read_borehole_table(self, data_dir):
        file_pq = data_dir / r"test_borehole_table.parquet"
        file_csv = data_dir / r"test_borehole_table.csv"
        cores_pq = read_borehole_table(file_pq)
        cores_csv = read_borehole_table(file_csv)
        assert isinstance(cores_pq, BoreholeCollection)
        assert isinstance(cores_csv, BoreholeCollection)

        cores_pq = read_borehole_table(file_pq, as_collection=False)
        cores_csv = read_borehole_table(file_csv, as_collection=False)
        assert isinstance(cores_pq, LayeredData)
        assert isinstance(cores_csv, LayeredData)

    @pytest.mark.unittest
    def test_read_inclined_borehole_table(self, data_dir):
        file_pq = data_dir / r"test_inclined_borehole_table.parquet"
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
    def test_adjust_z_coordinates(self, data_dir):
        file = data_dir / r"test_borehole_table.parquet"
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

    @pytest.mark.unittest
    def test_read_boris_xml(self, data_dir):
        boris_collection = read_xml_boris(data_dir / r"test_boris_xml.xml")
        assert isinstance(boris_collection, BoreholeCollection)
        assert boris_collection.n_points == 16
        assert len(boris_collection.data.df) == 236


@pytest.mark.unittest
def test_read_uullg_table(
    llg_header_table, llg_data_table, llg_data_table_duplicate_column
):
    llg = read_uullg_tables(llg_header_table, llg_data_table)
    assert isinstance(llg, BoreholeCollection)
    expected_header_cols = ["nr", "x", "y", "surface", "end"]
    header_columns = llg.header.gdf.columns
    assert all([c in header_columns for c in expected_header_cols])
    expected_data_cols = ["nr", "x", "y", "surface", "end"]
    data_columns = llg.data.df.columns
    assert all([c in data_columns for c in expected_data_cols])

    llg = read_uullg_tables(llg_header_table, llg_data_table_duplicate_column)
    assert isinstance(llg, BoreholeCollection)
    expected_header_cols = ["nr", "x", "y", "surface", "end"]
    header_columns = llg.header.gdf.columns
    assert all([c in header_columns for c in expected_header_cols])
    expected_data_cols = ["nr", "x", "y", "surface", "end"]
    data_columns = llg.data.df.columns
    assert all([c in data_columns for c in expected_data_cols])


@pytest.mark.unittest
def test_read_gef_cpts(data_dir):
    files = sorted(Path(data_dir / "cpt").glob("*.gef"))
    cpts = read_gef_cpts(files)
    assert isinstance(cpts, CptCollection)

    expected_cpts_present = [
        "DKMP_D03",
        "AZZ158",
        "CPT000000157983",
        "YANGTZEHAVEN CPT 10",
    ]
    assert_array_equal(cpts.header["nr"], expected_cpts_present)


@pytest.mark.unittest
def test_read_cpt_table(data_dir, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "mv")
    cpts = read_cpt_table(data_dir / r"test_cpts.parquet")
    assert isinstance(cpts, CptCollection)
    assert cpts.horizontal_reference == 28992
    assert cpts.vertical_reference == 5709

    cpts = read_cpt_table(data_dir / r"test_cpts.parquet", as_collection=False)
    assert isinstance(cpts, DiscreteData)


@pytest.mark.unittest
def test_read_pickle(collection_pickle):
    collection = read_pickle(collection_pickle)
    assert isinstance(collection, BoreholeCollection)


class TestReadCollectionGeopackage:
    @pytest.fixture
    def borehole_gpkg(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"borehole.gpkg"
        borehole_collection.to_geopackage(outfile)
        return outfile

    @pytest.fixture
    def cpt_gpkg(self, cpt_collection, tmp_path):
        outfile = tmp_path / r"cpt.gpkg"
        cpt_collection.to_geopackage(outfile)
        return outfile

    @pytest.mark.unittest
    def test_read_collection_geopackage_boreholes(self, borehole_gpkg):
        collection = read_collection_geopackage(borehole_gpkg, BoreholeCollection)
        assert isinstance(collection, BoreholeCollection)
        assert isinstance(collection.header, PointHeader)
        assert isinstance(collection.data, LayeredData)

    @pytest.mark.unittest
    def test_read_collection_geopackage_cpts(self, cpt_gpkg):
        collection = read_collection_geopackage(cpt_gpkg, CptCollection)
        assert isinstance(collection, CptCollection)
        assert isinstance(collection.header, PointHeader)
        assert isinstance(collection.data, DiscreteData)

    @pytest.mark.unittest
    def test_read_collection_geopackage_invalid(self, borehole_gpkg):
        with pytest.raises(ValueError):
            read_collection_geopackage(borehole_gpkg, InvalidCollection)
