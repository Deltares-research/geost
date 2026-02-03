import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely import geometry as gmt

import geost
from geost.accessors.data import DiscreteData, LayeredData
from geost.accessors.header import PointHeader
from geost.base import (
    BoreholeCollection,
    CptCollection,
)
from geost.io.read import (
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


@pytest.fixture
def table_wrong_columns():
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
def test_nlog_reader_from_parquet(testdatadir):
    nlog_cores = geost.read_nlog_cores(
        testdatadir / r"test_nlog_stratstelsel_20230807.parquet"
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
    assert nlog_cores.has_inclined


@pytest.mark.unittest
def test_read_borehole_table(testdatadir):
    file_pq = testdatadir / r"test_borehole_table.parquet"
    file_csv = testdatadir / r"test_borehole_table.csv"
    cores_pq = geost.read_borehole_table(file_pq)
    cores_csv = geost.read_borehole_table(file_csv)
    assert isinstance(cores_pq, BoreholeCollection)
    assert isinstance(cores_csv, BoreholeCollection)

    cores_pq = geost.read_borehole_table(file_pq, as_collection=False)
    cores_csv = geost.read_borehole_table(file_csv, as_collection=False)
    assert isinstance(cores_pq, pd.DataFrame)
    assert isinstance(cores_csv, pd.DataFrame)

    file_pq = testdatadir / r"test_inclined_borehole_table.parquet"
    cores_pq = geost.read_borehole_table(file_pq, has_inclined=True)
    assert isinstance(cores_pq, BoreholeCollection)
    assert cores_pq.has_inclined

    cores_pq = geost.read_borehole_table(
        file_pq, has_inclined=True, as_collection=False
    )
    assert isinstance(cores_pq, pd.DataFrame)


@pytest.mark.unittest
def test_check_mandatory_columns(table_wrong_columns):
    column_mapper = {"maaiveld": "surface"}
    table_wrong_columns = _check_mandatory_column_presence(
        table_wrong_columns, MANDATORY_LAYERED_DATA_COLUMNS, column_mapper
    )
    assert_array_equal(table_wrong_columns.columns, MANDATORY_LAYERED_DATA_COLUMNS)


@pytest.mark.unittest
def test_check_mandatory_columns_with_user_input(table_wrong_columns, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "maaiveld")
    table_wrong_columns = _check_mandatory_column_presence(
        table_wrong_columns, MANDATORY_LAYERED_DATA_COLUMNS
    )
    assert_array_equal(table_wrong_columns.columns, MANDATORY_LAYERED_DATA_COLUMNS)


@pytest.mark.unittest
def test_adjust_z_coordinates(testdatadir):
    file = testdatadir / r"test_borehole_table.parquet"
    cores = geost.read_borehole_table(file, as_collection=False)
    test_df = cores.copy()

    # test situation where top and bottoms are already positive downward: no change
    test_df = adjust_z_coordinates(test_df)
    assert test_df.equals(cores)
    assert test_df["top"].iloc[0] < test_df["top"].iloc[1]
    assert test_df["bottom"].iloc[0] < test_df["bottom"].iloc[1]

    # Test situation where top and bottoms are given as negative downward
    test_df = cores.copy()
    test_df["top"] *= -1
    test_df["bottom"] *= -1

    test_df = adjust_z_coordinates(test_df)
    assert test_df.equals(cores)
    assert test_df["top"].iloc[0] < test_df["top"].iloc[1]
    assert test_df["bottom"].iloc[0] < test_df["bottom"].iloc[1]


@pytest.mark.unittest
def test_read_boris_xml(testdatadir):
    boris_collection = geost.read_xml_boris(testdatadir / r"xml/test_boris_xml.xml")
    assert isinstance(boris_collection, BoreholeCollection)
    assert boris_collection.n_points == 16
    assert len(boris_collection.data) == 236


@pytest.mark.unittest
def test_read_uullg_table(
    llg_header_table, llg_data_table, llg_data_table_duplicate_column
):
    llg = geost.read_uullg_tables(llg_header_table, llg_data_table)
    assert isinstance(llg, BoreholeCollection)
    expected_header_cols = ["nr", "x", "y", "surface", "end"]
    header_columns = llg.header.columns
    assert all([c in header_columns for c in expected_header_cols])
    expected_data_cols = ["nr", "x", "y", "surface", "end"]
    data_columns = llg.data.columns
    assert all([c in data_columns for c in expected_data_cols])

    llg = geost.read_uullg_tables(llg_header_table, llg_data_table_duplicate_column)
    assert isinstance(llg, BoreholeCollection)
    expected_header_cols = ["nr", "x", "y", "surface", "end"]
    header_columns = llg.header.columns
    assert all([c in header_columns for c in expected_header_cols])
    expected_data_cols = ["nr", "x", "y", "surface", "end"]
    data_columns = llg.data.columns
    assert all([c in data_columns for c in expected_data_cols])


@pytest.mark.unittest
def test_read_gef_cpts(testdatadir):
    files = sorted(Path(testdatadir / "gef").glob("*.gef"))
    cpts = geost.read_gef_cpts(files)
    assert isinstance(cpts, CptCollection)
    assert cpts.horizontal_reference == 28992
    assert cpts.vertical_reference == 5709

    expected_cpts_present = [
        "DKMP_D03",
        "AZZ158",
        "CPT000000038871",
        "CPT000000157983",
        "YANGTZEHAVEN CPT 10",
    ]
    assert_array_equal(cpts.header["nr"], expected_cpts_present)


@pytest.mark.unittest
def test_read_cpt_table(testdatadir, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "mv")
    cpts = geost.read_cpt_table(testdatadir / r"test_cpts.parquet")
    assert isinstance(cpts, CptCollection)
    assert cpts.horizontal_reference == 28992
    assert cpts.vertical_reference == 5709

    cpts = geost.read_cpt_table(testdatadir / r"test_cpts.parquet", as_collection=False)
    assert isinstance(cpts, pd.DataFrame)


@pytest.mark.unittest
def test_read_pickle(collection_pickle):
    collection = geost.read_pickle(collection_pickle)
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
        collection = geost.read_collection_geopackage(borehole_gpkg, BoreholeCollection)
        assert isinstance(collection, BoreholeCollection)
        assert isinstance(collection.header, gpd.GeoDataFrame)
        assert isinstance(collection.data, pd.DataFrame)
        assert not collection.has_inclined

    @pytest.mark.unittest
    def test_read_collection_geopackage_cpts(self, cpt_gpkg):
        collection = geost.read_collection_geopackage(cpt_gpkg, CptCollection)
        assert isinstance(collection, CptCollection)
        assert isinstance(collection.header, gpd.GeoDataFrame)
        assert isinstance(collection.data, pd.DataFrame)
        assert not collection.has_inclined

    @pytest.mark.unittest
    def test_read_collection_geopackage_invalid(self, borehole_gpkg):
        with pytest.raises(ValueError):
            geost.read_collection_geopackage(borehole_gpkg, InvalidCollection)


@pytest.mark.parametrize(
    "object_type, options, expected_bro_ids",
    [
        (
            "BHR-GT",
            {"bro_ids": ["BHR000000339682", "BHR000000339733"]},
            ["BHR000000339682", "BHR000000339733"],
        ),
        (
            "BHR-GT",
            {"bbox": (132780.327, 448030.0, 132782.327, 448032.1)},
            ["BHR000000336600"],
        ),
        (
            "BHR-GT",
            {"geometry": gmt.box(132780.327, 448030.0, 132782.327, 448032.1)},
            ["BHR000000336600"],
        ),
        (
            "BHR-GT",
            {
                "geometry": gpd.GeoDataFrame(
                    [None],
                    geometry=[
                        gmt.Point(641589.8677899896, 5765293.564203509),
                    ],
                    crs=32631,
                ),
                "buffer": 2,
            },
            ["BHR000000336600"],
        ),
        (
            "BHR-P",
            {"bro_ids": "BHR000000108193"},
            ["BHR000000108193"],
        ),
        (
            "BHR-P",
            {"bbox": (129490, 452254, 129492, 452256)},
            ["BHR000000108193"],
        ),
        (
            "BHR-P",
            {"geometry": gmt.box(129490, 452254, 129492, 452256)},
            ["BHR000000108193"],
        ),
        (
            "BHR-P",
            {
                "geometry": gpd.GeoDataFrame(
                    [None],
                    geometry=[
                        gmt.Point(638162.819314087, 5769406.789567029),
                    ],
                    crs=32631,
                ),
                "buffer": 2,
            },
            ["BHR000000108193"],
        ),
        (
            "BHR-G",
            {"bro_ids": "BHR000000396406"},
            ["BHR000000396406"],
        ),
        (
            "BHR-G",
            {"bbox": (126148, 452161, 126150, 452163)},
            ["BHR000000396406"],
        ),
        (
            "BHR-G",
            {"geometry": gmt.box(126148, 452161, 126150, 452163)},
            ["BHR000000396406"],
        ),
        (
            "BHR-G",
            {
                "geometry": gpd.GeoDataFrame(
                    [None],
                    geometry=[
                        gmt.Point(634825.970503354, 5769204.047473239),
                    ],
                    crs=32631,
                ),
                "buffer": 2,
            },
            ["BHR000000396406"],
        ),
        (
            "CPT",
            {"bro_ids": "CPT000000155283"},
            ["CPT000000155283"],
        ),
        (
            "CPT",
            {"bbox": (132781.52, 448029.34, 132783.52, 448031.34)},
            ["CPT000000155283"],
        ),
        (
            "CPT",
            {"geometry": gmt.box(132781.52, 448029.34, 132783.52, 448031.34)},
            ["CPT000000155283"],
        ),
        (
            "CPT",
            {
                "geometry": gpd.GeoDataFrame(
                    [None],
                    geometry=[
                        gmt.Point(641591.0850222938, 5765292.843843447),
                    ],
                    crs=32631,
                ),
                "buffer": 2,
            },
            ["CPT000000155283"],
        ),
        (
            "SFR",
            {"bro_ids": "SFR000000000687"},
            ["SFR000000000687"],
        ),
        (
            "SFR",
            {"bbox": (132249, 451074, 132251, 451076)},
            ["SFR000000000687"],
        ),
        (
            "SFR",
            {"geometry": gmt.box(132249, 451074, 132251, 451076)},
            ["SFR000000000687"],
        ),
        (
            "SFR",
            {
                "geometry": gpd.GeoDataFrame(
                    [None],
                    geometry=[
                        gmt.Point(640958.8869981834, 5768318.157526063),
                    ],
                    crs=32631,
                ),
                "buffer": 2,
            },
            ["SFR000000000687"],
        ),
    ],
)
@pytest.mark.unittest
def test_bro_api_read(object_type, options, expected_bro_ids):
    collection = geost.bro_api_read(object_type, **options)
    assert isinstance(collection, geost.base.Collection)
    assert len(collection) == len(expected_bro_ids)
    assert_array_equal(collection.header["nr"], expected_bro_ids)


@pytest.mark.parametrize(
    "object_type",
    [
        "BHR-GT",
        "BHR-G",
        "CPT",
        "BHR-P",
        "SFR",
    ],
)
@pytest.mark.unittest
def test_bro_api_read_no_results(object_type):
    collection = geost.bro_api_read(object_type, bbox=[0, 0, 1, 1])
    assert isinstance(collection, geost.base.Collection)
    assert "<EMPTY COLLECTION>" in str(collection)


@pytest.mark.unittest
def test_bro_api_read_invalid_object_type():
    with pytest.raises(
        ValueError, match="Object type 'INVALID' is not supported for reading."
    ):
        geost.bro_api_read("INVALID", bro_ids=["BHR000000339682"])


@pytest.mark.unittest
def test_read_bhrp(testdatadir):
    file = testdatadir / r"xml/bhrp_bro.xml"
    collection = geost.read_bhrp(file)
    assert isinstance(collection, BoreholeCollection)


@pytest.mark.unittest
def test_read_bhrgt(testdatadir):
    file = testdatadir / r"xml/bhrgt_bro.xml"
    collection = geost.read_bhrgt(file)
    assert isinstance(collection, BoreholeCollection)


@pytest.mark.unittest
def test_read_bhrg(testdatadir):
    file = testdatadir / r"xml/bhrg_bro.xml"
    collection = geost.read_bhrg(file)
    assert isinstance(collection, BoreholeCollection)


@pytest.mark.unittest
def test_read_cpt(testdatadir):
    file = testdatadir / r"xml/cpt_bro.xml"
    collection = geost.read_cpt(file)
    assert isinstance(collection, CptCollection)


@pytest.mark.unittest
def test_read_sfr(testdatadir):
    file = testdatadir / r"xml/sfr_bro.xml"
    collection = geost.read_sfr(file)
    assert isinstance(collection, BoreholeCollection)
