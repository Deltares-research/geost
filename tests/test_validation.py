import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geost import BoreholeCollection, config
from geost._warnings import AlignmentWarning, ValidationWarning


@pytest.fixture
def valid_boreholes():
    nr = np.full(10, "B-01")
    x = np.full(10, 139370)
    y = np.full(10, 455540)
    mv = np.full(10, 1.0)
    end = np.full(10, -4.0)
    top = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
    bottom = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8])
    data_string = np.array(["K", "Kz", "K", "Ks3", "Ks2", "V", "Zk", "Zs", "Z", "Z"])
    data_int = np.arange(0, 10, dtype=np.int64)
    data_float = np.arange(0, 5, 0.5, dtype=np.float64)

    return pd.DataFrame(
        {
            "nr": nr,
            "x": x,
            "y": y,
            "surface": mv,
            "end": end,
            "top": top,
            "bottom": bottom,
            "data_string": data_string,
            "data_int": data_int,
            "data_float": data_float,
        }
    )


@pytest.fixture
def invalid_borehole_table():
    nr = np.full(10, 1)
    x = np.full(10, 139370)
    y = np.full(10, 455540)
    mv = np.full(10, 1.0)
    end = np.full(10, 4.0)
    top = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
    bottom = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 6.5])
    data_string = np.array(["K", "Kz", "K", "Ks3", "Ks2", "V", "Zk", "Zs", "Z", "Z"])
    data_int = np.arange(0, 10, dtype=np.int64)
    data_float = np.arange(0, 5, 0.5, dtype=np.float64)

    return pd.DataFrame(
        {
            "nr": nr,
            "x": x,
            "y": y,
            "surface": mv,
            "end": end,
            "top": top,
            "bottom": bottom,
            "data_string": data_string,
            "data_int": data_int,
            "data_float": data_float,
        }
    )


@pytest.fixture
def misaligned_header():
    return gpd.GeoDataFrame(
        {
            "nr": ["B-02"],
            "x": [100000],
            "y": [400000],
            "surface": [0],
            "end": [-8],
        },
        geometry=[Point(100000, 400000)],
        crs=28992,
    )


@pytest.mark.unittest
def test_validation_pass(valid_boreholes):
    config.validation.reset_settings()  # Ensure default settings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        valid_boreholes.gstda.to_collection()


@pytest.mark.unittest
def test_validation_fail(invalid_borehole_table):
    config.validation.reset_settings()  # Ensure default settings
    config.validation.VERBOSE = True
    config.validation.DROP_INVALID = False
    config.validation.FLAG_INVALID = False

    with pytest.warns(ValidationWarning) as record:
        invalid_borehole_table.gstda.to_collection()

    assert len(record) == 2

    header_result, data_result = record
    assert "Validation failed for schema 'Point header'" in str(header_result.message)
    assert (
        "DataFrameSchema 'Point header' failed element-wise validator number 0:"
        in str(header_result.message)
    )
    assert "Validation failed for schema 'Layer data non-inclined'" in str(
        data_result.message
    )
    assert (
        "DataFrameSchema 'Layer data non-inclined' failed element-wise validator number 0:"
        in str(data_result.message)
    )


@pytest.mark.unittest
def test_validation_fail_drop_invalid(invalid_borehole_table):
    config.validation.reset_settings()  # Ensure default settings
    config.validation.VERBOSE = True
    config.validation.DROP_INVALID = True
    config.validation.FLAG_INVALID = False

    with pytest.warns(ValidationWarning) as record:
        invalid_borehole_table.gstda.to_collection()

    assert len(record) == 2

    assert "Validation dropped 1 row(s) for schema 'Point header'" in str(
        record[0].message
    )
    assert "Validation dropped 10 row(s) for schema 'Layer data non-inclined'" in str(
        record[1].message
    )


@pytest.mark.unittest
def test_validation_fail_flag_invalid(invalid_borehole_table):
    config.validation.reset_settings()  # Ensure default settings
    config.validation.VERBOSE = True
    config.validation.DROP_INVALID = False
    config.validation.FLAG_INVALID = True

    with pytest.warns(ValidationWarning) as record:
        collection_flagged = invalid_borehole_table.gstda.to_collection()

    assert len(record) == 2
    assert not collection_flagged.header.loc[0, "is_valid"]
    assert not collection_flagged.data.loc[9, "is_valid"]

    header_result, data_result = record
    assert "Validation failed for schema 'Point header'" in str(header_result.message)
    assert (
        "DataFrameSchema 'Point header' failed element-wise validator number 0:"
        in str(header_result.message)
    )
    assert "Validation failed for schema 'Layer data non-inclined'" in str(
        data_result.message
    )
    assert (
        "DataFrameSchema 'Layer data non-inclined' failed element-wise validator number 0:"
        in str(data_result.message)
    )


@pytest.mark.unittest
def test_header_mismatch_auto_align(valid_boreholes, misaligned_header):
    config.validation.reset_settings()  # Ensure default settings
    config.validation.AUTO_ALIGN = True

    with pytest.warns(AlignmentWarning) as record:
        BoreholeCollection(misaligned_header, valid_boreholes)

    assert len(record) == 2
    assert "Header covers more/other objects than present in the data table" in str(
        record[0].message
    )
    assert "Header does not cover all unique objects in data" in str(record[1].message)


@pytest.mark.unittest
def test_header_mismatch_no_auto_align(valid_boreholes, misaligned_header):
    config.validation.reset_settings()  # Ensure default settings
    config.validation.AUTO_ALIGN = False

    with pytest.warns(AlignmentWarning) as record:
        BoreholeCollection(misaligned_header, valid_boreholes)

    assert len(record) == 2
    assert "Header covers more/other objects than present in the data table" in str(
        record[0].message
    )
    assert "Header does not cover all unique objects in data" in str(record[1].message)


@pytest.mark.unittest
def test_validation_skipped(invalid_borehole_table):
    config.validation.reset_settings()  # Ensure default settings
    config.validation.SKIP = True

    collection = invalid_borehole_table.gstda.to_collection()
    assert collection.data.equals(
        invalid_borehole_table
    )  # No validation applied, nothin changed
