import pandas as pd
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
)

from geost.analysis.grainsize import (
    calculate_bhrgt_grainsize_fractions,
    calculate_bhrgt_grainsize_percentiles,
)
from geost.base import Collection


@pytest.fixture
def bhrgt_samples_data():
    df = pd.DataFrame(
        {
            "nr": ["B01", "B01", "B01"],
            "x": [619_500, 619_500, 619_500],
            "y": [5_924_500, 5_924_500, 5_924_500],
            "surface": [-20.0, -20.0, -20.0],
            "end": [-23.0, -23.0, -23.0],
            "top": [0.4, 1.4, 2.4],
            "bottom": [0.6, 2.6, 3.6],
            "fractionSmaller63um": [10.0, 20.0, 30.0],
            "fractionLarger63um": [90.0, 80.0, 70.0],
            "fraction63to90um": [5.0, 30.0, 40.0],
            "fraction90to125um": [10.0, 15.0, 10.0],
            "fraction125to180um": [15.0, 15.0, 10.0],
            "fraction180to250um": [20.0, 10.0, 5.0],
            "fraction250to355um": [10.0, 5.0, 5.0],
            "fraction355to500um": [10.0, 5.0, 0.0],
            "fraction500to710um": [8.0, 0.0, 0.0],
            "fraction710to1000um": [4.0, 0.0, 0.0],
            "fraction1000to1400um": [3.0, 0.0, 0.0],
            "fraction1400umto2mm": [2.0, 0.0, 0.0],
            "fraction2to4mm": [2.0, 0.0, 0.0],
            "fraction4to8mm": [1.0, 0.0, 0.0],
            "fraction8to16mm": [0.0, 0.0, 0.0],
            "fraction16to31_5mm": [0.0, 0.0, 0.0],
            "fraction31_5to63mm": [0.0, 0.0, 0.0],
            "fractionLarger63mm": [0.0, 0.0, 0.0],
        }
    )
    return df


@pytest.fixture
def bhrgt_samples_collection(bhrgt_samples_data):
    return bhrgt_samples_data.gst.to_collection(
        crs=32631, include_in_header=["nr", "x", "y", "surface", "end"]
    )


@pytest.mark.unittest
def test_calculate_bhrgt_grainsize_percentiles(
    bhrgt_samples_data, bhrgt_samples_collection
):
    result_df = calculate_bhrgt_grainsize_percentiles(
        bhrgt_samples_data, percentiles=[10, 50, 90], only_sand=True
    )
    assert_array_equal(
        result_df.columns,
        list(bhrgt_samples_data.columns) + ["d10_sand", "d50_sand", "d90_sand"],
    )
    assert_array_almost_equal(result_df["d10_sand"], [102.95, 70.2, 67.725], decimal=2)
    assert_array_almost_equal(
        result_df["d50_sand"], [227.25, 113.33, 86.625], decimal=2
    )
    assert_array_almost_equal(result_df["d90_sand"], [731.75, 292.0, 222.0], decimal=2)

    result_df = calculate_bhrgt_grainsize_percentiles(
        bhrgt_samples_data, percentiles=[10, 50, 90], only_sand=False
    )
    assert_array_almost_equal(result_df["d10"], [63.0, 31.5, 21.0], decimal=2)
    assert_array_almost_equal(result_df["d50"], [215.0, 90.0, 76.5], decimal=2)
    assert_array_almost_equal(result_df["d90"], [855.0, 250.0, 180.0], decimal=2)

    # Test run implementation for BoreholeCollection and percentile as integer.
    result_collection = calculate_bhrgt_grainsize_percentiles(
        bhrgt_samples_collection, percentiles=50, only_sand=True
    )
    assert isinstance(result_collection, Collection)


@pytest.mark.unittest
def test_calculate_bhrgt_grainsize_fractions(
    bhrgt_samples_data, bhrgt_samples_collection
):
    result_df = calculate_bhrgt_grainsize_fractions(bhrgt_samples_data)
    assert_array_almost_equal(result_df["perc_fines"], [10.0, 20.0, 30.0], decimal=2)
    assert_array_almost_equal(result_df["perc_sand"], [87.0, 80.0, 70.0], decimal=2)
    assert_array_almost_equal(result_df["perc_gravel"], [3.0, 0.0, 0.0], decimal=2)

    # Test run implementation for BoreholeCollection without assertions
    result_collection = calculate_bhrgt_grainsize_fractions(bhrgt_samples_collection)
    assert isinstance(result_collection, Collection)
