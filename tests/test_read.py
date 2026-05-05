import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely import geometry as gmt

import geost
from geost.base import Collection


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


@pytest.mark.unittest
def test_nlog_reader_from_parquet(testdatadir):
    nlog = geost.read_nlog_cores(
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
    assert isinstance(nlog, Collection)
    assert nlog.header[desired_df.columns].equals(desired_df)
    assert nlog.has_inclined
    assert nlog.crs == 28992
    assert nlog.vertical_datum == 5709
    assert nlog.data.shape == (62, 29)


@pytest.mark.parametrize(
    "filename",
    [
        "test_borehole_table.parquet",
        "test_borehole_table.csv",
        "test_inclined_borehole_table.parquet",
    ],
)
def test_read_borehole_table(filename, testdatadir):
    filepath = testdatadir / filename
    if filename == "test_inclined_borehole_table.parquet":
        cores = geost.read_borehole_table(filepath, coll_kwargs=dict(has_inclined=True))
        assert cores.has_inclined
    else:
        cores = geost.read_borehole_table(filepath)
        assert not cores.has_inclined
    assert isinstance(cores, Collection)

    cores = geost.read_borehole_table(filepath, as_collection=False)
    assert isinstance(cores, pd.DataFrame)


@pytest.mark.unittest
def test_read_boris_xml(testdatadir):
    collection = geost.read_xml_boris(
        testdatadir / r"xml/test_boris_xml.xml",
        crs=28992,
        include_in_header=["nr", "x", "y", "surface", "end"],
        has_inclined=False,
    )
    assert isinstance(collection, Collection)
    assert isinstance(collection.header, gpd.GeoDataFrame)
    assert isinstance(collection.data, pd.DataFrame)
    assert_array_equal(
        collection.header.columns, ["nr", "x", "y", "surface", "end", "geometry"]
    )
    assert collection.header.shape == (16, 6)
    assert collection.data.shape == (236, 51)
    assert collection.crs == 28992
    assert collection.vertical_datum is None
    assert not collection.has_inclined

    df = geost.read_xml_boris(
        testdatadir / r"xml/test_boris_xml.xml", as_collection=False
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (236, 51)


@pytest.mark.unittest
def test_read_uullg_table(
    llg_header_table, llg_data_table, llg_data_table_duplicate_column
):
    llg = geost.read_uullg_tables(llg_header_table, llg_data_table)
    assert isinstance(llg, Collection)
    expected_header_cols = ["nr", "x", "y", "surface", "end"]
    header_columns = llg.header.columns
    assert all([c in header_columns for c in expected_header_cols])
    expected_data_cols = ["nr", "x", "y", "surface", "end"]
    data_columns = llg.data.columns
    assert all([c in data_columns for c in expected_data_cols])

    llg = geost.read_uullg_tables(llg_header_table, llg_data_table_duplicate_column)
    assert isinstance(llg, Collection)
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
    assert isinstance(cpts, Collection)
    assert cpts.crs is None
    assert cpts.vertical_datum is None
    assert_array_equal(
        cpts.header["nr"],
        [
            "DKMP_D03",
            "AZZ158",
            "CPT000000038871",
            "CPT000000157983",
            "YANGTZEHAVEN CPT 10",
        ],
    )

    cpts_df = geost.read_gef_cpts(files, as_collection=False)
    assert isinstance(cpts_df, pd.DataFrame)


@pytest.mark.unittest
def test_read_cpt_table(testdatadir, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "mv")
    cpts = geost.read_cpt_table(testdatadir / r"test_cpts.parquet")
    assert isinstance(cpts, Collection)
    assert cpts.crs is None
    assert cpts.vertical_datum is None

    cpts = geost.read_cpt_table(testdatadir / r"test_cpts.parquet", as_collection=False)
    assert isinstance(cpts, pd.DataFrame)


@pytest.mark.unittest
def test_read_pickle(collection_pickle):
    collection = geost.read_pickle(collection_pickle)
    assert isinstance(collection, Collection)


@pytest.mark.parametrize("collection", ["borehole_collection", "cpt_collection"])
def test_read_collection_geopackage(collection, tmp_path, request):
    outfile = tmp_path / r"collection.gpkg"
    collection = request.getfixturevalue(collection)
    collection.to_geopackage(outfile)
    read_collection = geost.read_collection_geopackage(outfile, Collection)
    assert isinstance(read_collection, Collection)
    assert isinstance(read_collection.header, gpd.GeoDataFrame)
    assert isinstance(read_collection.data, pd.DataFrame)
    assert read_collection.header.equals(collection.header)
    assert read_collection.data.equals(collection.data)


@pytest.mark.parametrize(
    "object_type, options, expected_bro_ids, expected_warning",
    [
        (
            "BHR-GT",
            {"bro_ids": ["BHR000000339682", "BHR000000339733"]},
            ["BHR000000339682", "BHR000000339733"],
            None,
        ),
        (
            "BHR-GT",
            {"bbox": (132780.327, 448030.0, 132782.327, 448032.1)},
            ["BHR000000336600"],
            None,
        ),
        (
            "BHR-GT",
            {"geometry": gmt.box(132780.327, 448030.0, 132782.327, 448032.1)},
            ["BHR000000336600"],
            UserWarning,
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
            None,
        ),
        (
            "BHR-GT",
            {
                "geometry": gpd.GeoDataFrame(
                    [None],
                    geometry=[
                        gmt.Point(641589.8677899896, 5765293.564203509),
                    ],
                    crs=None,
                ),
                "buffer": 2,
                "crs": 32631,
            },
            ["BHR000000336600"],
            UserWarning,
        ),
        (
            "BHR-P",
            {"bro_ids": "BHR000000108193"},
            ["BHR000000108193"],
            None,
        ),
        (
            "BHR-P",
            {"bbox": (129490, 452254, 129492, 452256)},
            ["BHR000000108193"],
            None,
        ),
        (
            "BHR-P",
            {"geometry": gmt.box(129490, 452254, 129492, 452256)},
            ["BHR000000108193"],
            UserWarning,
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
            None,
        ),
        (
            "BHR-G",
            {"bro_ids": "BHR000000396406"},
            ["BHR000000396406"],
            None,
        ),
        (
            "BHR-G",
            {"bbox": (126148, 452161, 126150, 452163)},
            ["BHR000000396406"],
            None,
        ),
        (
            "BHR-G",
            {"geometry": gmt.box(126148, 452161, 126150, 452163)},
            ["BHR000000396406"],
            UserWarning,
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
            None,
        ),
        (
            "CPT",
            {"bro_ids": "CPT000000155283"},
            ["CPT000000155283"],
            None,
        ),
        (
            "CPT",
            {"bbox": (132781.52, 448029.34, 132783.52, 448031.34)},
            ["CPT000000155283"],
            None,
        ),
        (
            "CPT",
            {"geometry": gmt.box(132781.52, 448029.34, 132783.52, 448031.34)},
            ["CPT000000155283"],
            UserWarning,
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
            None,
        ),
        (
            "SFR",
            {"bro_ids": "SFR000000000687"},
            ["SFR000000000687"],
            None,
        ),
        (
            "SFR",
            {"bbox": (132249, 451074, 132251, 451076)},
            ["SFR000000000687"],
            None,
        ),
        (
            "SFR",
            {"geometry": gmt.box(132249, 451074, 132251, 451076)},
            ["SFR000000000687"],
            UserWarning,
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
            None,
        ),
    ],
)
@pytest.mark.unittest
def test_bro_api_read(object_type, options, expected_bro_ids, expected_warning):
    if expected_warning is not None:
        with warnings.catch_warnings(record=True) as w:
            collection = geost.bro_api_read(object_type, **options)
            assert len(w) == 1
            assert issubclass(w[-1].category, expected_warning)
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
    assert collection.header.empty
    assert collection.data.empty


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
    assert isinstance(collection, Collection)


@pytest.mark.unittest
def test_read_bhrgt(testdatadir):
    file = testdatadir / r"xml/bhrgt_bro.xml"
    collection = geost.read_bhrgt(file)
    assert isinstance(collection, Collection)


@pytest.mark.unittest
def test_read_bhrgt_samples(testdatadir):
    file = testdatadir / r"xml/bhrgt_bro_with_samples.xml"
    collection = geost.read_bhrgt_samples(file)
    assert isinstance(collection, Collection)


@pytest.mark.unittest
def test_read_bhrg(testdatadir):
    file = testdatadir / r"xml/bhrg_bro.xml"
    collection = geost.read_bhrg(file)
    assert isinstance(collection, Collection)


@pytest.mark.unittest
def test_read_cpt(testdatadir):
    file = testdatadir / r"xml/cpt_bro.xml"
    collection = geost.read_cpt(file)
    assert isinstance(collection, Collection)


@pytest.mark.unittest
def test_read_sfr(testdatadir):
    file = testdatadir / r"xml/sfr_bro.xml"
    collection = geost.read_sfr(file)
    assert isinstance(collection, Collection)
