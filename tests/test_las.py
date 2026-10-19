import lasio
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost.io.las import (
    standardize_well_log_las,
    well_logs_to_collection,
)


@pytest.mark.unittest
def test_las_header_contains_unit_columns():
    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem(
        mnemonic="WELL", value="BH-001", unit="", descr="Well ID"
    )
    las.well["STRT"] = lasio.HeaderItem(
        mnemonic="STRT", value=0.0, unit="m", descr="start"
    )
    las.well["STOP"] = lasio.HeaderItem(
        mnemonic="STOP", value=100.0, unit="m", descr="stop"
    )
    las.well["EGL"] = lasio.HeaderItem(
        mnemonic="EGL", value=10.0, unit="m", descr="surface"
    )
    las.curves.append(lasio.CurveItem(mnemonic="DEPTH", unit="m", descr="1  DEPTH"))
    las.curves.append(lasio.CurveItem(mnemonic="GR", unit="gAPI", descr="2  Gamma ray"))
    las.set_data([[0.0, 50.0], [1.0, 60.0]])

    standardized = standardize_well_log_las(las, errors="raise")
    collection = well_logs_to_collection(standardized)

    assert {
        "nr",
        "strt",
        "strt_unit",
        "stop",
        "stop_unit",
        "surface",
        "surface_unit",
        "depth",
        "depth_unit",
        "gamma",
        "gamma_unit",
    }.issubset(collection.header.columns)

    assert {
        "depth",
        "gamma",
    }.issubset(collection.data.columns)

    assert_array_equal(collection.data["depth"].values, [0.0, 1.0])
    assert_array_equal(collection.data["gamma"].values, [50.0, 60.0])

    assert_array_equal(
        collection.header.loc[0, ["depth", "depth_unit", "gamma", "gamma_unit"]],
        ["DEPTH", "m", "Gamma ray", "gAPI"],
    )


@pytest.mark.unittest
def test_standardize_well_log_accepts_single_file():
    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem(mnemonic="WELL", value="BH-003")
    las.curves.append(lasio.CurveItem(mnemonic="DEPTH", unit="m"))
    las.set_data([[0.0]])

    standardized = standardize_well_log_las(las)

    assert isinstance(standardized, lasio.LASFile)
    assert standardized.well["nr"].value == "BH-003"


@pytest.mark.unittest
def test_las_header_skips_empty_well_and_parameter_items():
    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem(mnemonic="WELL", value="BH-004")
    las.well["EMPTY_WELL"] = lasio.HeaderItem(mnemonic="EMPTY_WELL", value="", unit="")
    las.params["EMPTY_PARAM"] = lasio.HeaderItem(
        mnemonic="EMPTY_PARAM", value="", unit=""
    )
    las.params["PARAM_WITH_UNIT"] = lasio.HeaderItem(
        mnemonic="PARAM_WITH_UNIT", value="", unit="m"
    )
    las.params["PARAM_WITH_VALUE"] = lasio.HeaderItem(
        mnemonic="PARAM_WITH_VALUE", value="present", unit=""
    )
    las.curves.append(lasio.CurveItem(mnemonic="DEPTH", unit="m", descr="depth"))
    las.set_data([[0.0]])

    standardized = standardize_well_log_las(las)
    collection = well_logs_to_collection(standardized)

    assert "empty_well" not in collection.header.columns
    assert "empty_param" not in collection.header.columns
    assert "PARAM_WITH_UNIT" not in collection.header.columns
    assert "PARAM_WITH_UNIT_unit" not in collection.header.columns
    assert "PARAM_WITH_VALUE" in collection.header.columns
    assert "PARAM_WITH_VALUE_unit" not in collection.header.columns


@pytest.mark.unittest
def test_las_header_includes_all_columns():
    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem(
        mnemonic="WELL", value="BH-002", unit="", descr="Well ID"
    )
    las.well["STRT"] = lasio.HeaderItem(
        mnemonic="STRT", value=0.0, unit="m", descr="start"
    )
    las.well["STOP"] = lasio.HeaderItem(
        mnemonic="STOP", value=100.0, unit="m", descr="stop"
    )
    las.well["EGL"] = lasio.HeaderItem(
        mnemonic="EGL", value=10.0, unit="m", descr="surface"
    )
    las.curves.append(lasio.CurveItem(mnemonic="DEPTH", unit="m", descr="depth"))
    las.curves.append(lasio.CurveItem(mnemonic="GR", unit="gAPI", descr="Gamma ray"))
    las.set_data([[0.0, 50.0], [1.0, 60.0]])

    standardized = standardize_well_log_las(las, errors="raise")
    collection = well_logs_to_collection(standardized)

    assert "nr" in collection.header.columns
    assert "nr_unit" not in collection.header.columns
    assert "surface" in collection.header.columns
    assert "surface_unit" in collection.header.columns
    assert "depth" in collection.header.columns
    assert "depth_unit" in collection.header.columns
    assert "strt" in collection.header.columns
    assert "strt_unit" in collection.header.columns
    assert "stop" in collection.header.columns
    assert "stop_unit" in collection.header.columns
    assert "gamma" in collection.header.columns
    assert "gamma_unit" in collection.header.columns
