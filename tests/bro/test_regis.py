import numpy as np
import pandas as pd
import pytest
from lxml import etree
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.bro.regis import (
    RegisUnits,
    regis_units,
)
from geost.exceptions import MissingUnitError


@pytest.mark.xfail(
    reason="Will fail in CI because pooch data can only be found from main branch"
)
@pytest.mark.unittest
def test_regis_units():
    metadata = regis_units()
    assert isinstance(metadata, RegisUnits)
    assert isinstance(metadata.df, pd.DataFrame)
    assert {"strat", "lithok"}.issubset(metadata.df.columns)
    assert isinstance(metadata.unit, pd.Index)
    assert isinstance(metadata.description, pd.Series)
    assert isinstance(metadata.colors_rgb, pd.DataFrame)
    assert isinstance(metadata.regis_version, str)
    assert metadata.regis_version == "v02r2s3"


class TestRegisUnits:
    @pytest.mark.unittest
    def test_unit_property(self, metadata_regis):
        assert isinstance(metadata_regis.unit, pd.Index)
        assert metadata_regis.unit.name == "HYD_UNIT_CD"

    @pytest.mark.unittest
    def test_description_property(self, metadata_regis):
        assert isinstance(metadata_regis.description, pd.Series)
        assert metadata_regis.description.name == "DESCRIPTION"
        assert_array_equal(metadata_regis.description.index, metadata_regis.unit)

    @pytest.mark.unittest
    def test_colors_rgb(self, metadata_regis):
        assert isinstance(metadata_regis.colors_rgb, pd.DataFrame)
        assert_array_equal(metadata_regis.colors_rgb.index, metadata_regis.unit)
        assert_array_equal(
            metadata_regis.colors_rgb.columns, ["RED_DEC", "GREEN_DEC", "BLUE_DEC"]
        )

    @pytest.mark.unittest
    def test_check_version(self, metadata_regis, regis_small):
        assert metadata_regis.check_version_matches(regis_small) is True

        with pytest.raises(
            TypeError,
            match="The model version cannot be found in a DataArray of a REGIS II variable",
        ):
            metadata_regis.check_version_matches(regis_small["kD"])

        with pytest.warns(UserWarning, match="REGIS II version mismatch"):
            regis_small.attrs["title"] = "v01r5s1"
            assert metadata_regis.check_version_matches(regis_small) is False

    @pytest.mark.parametrize(
        "units",
        ["NUhlc", ["NUhlc", "NUBXz"]],
        ids=["single-value", "list-of-values"],
    )
    def test_select_unit(self, metadata_regis, units):
        selected = metadata_regis.select_unit(units)
        assert isinstance(selected, RegisUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert_array_equal(selected.unit, units)

    @pytest.mark.unittest
    def test_select_unit_with_regis(self, metadata_regis, regis_small):
        selected = metadata_regis.select_unit(regis_small)
        assert isinstance(selected, RegisUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.df.shape == (31, 7)
        assert {"strat", "lithok"}.issubset(selected.df.columns)
        assert_array_equal(selected.unit, regis_small.gst.z)

    @pytest.mark.unittest
    def test_select_unit_error(self, metadata_regis):
        non_existing_unit = "NONEXISTENT_UNIT"
        with pytest.raises(MissingUnitError):
            metadata_regis.select_unit(non_existing_unit)

    @pytest.mark.unittest
    def test_select_unit_contains_single_value(self, metadata_regis):
        selected = metadata_regis.select_unit_contains("NUOO")
        assert isinstance(selected, RegisUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert_array_equal(
            selected.unit,
            ["NUOOz", "NUOOz1", "NUOOk1", "NUOOz2", "NUOOc"],
        )

    @pytest.mark.parametrize(
        "substring",
        [["NUOO", "NUDN"], np.array(["NUOO", "NUDN"]), pd.Series(["NUOO", "NUDN"])],
        ids=["list-of-values", "numpy-array", "pandas-series"],
    )
    def test_select_unit_contains_list_of_values(self, metadata_regis, substring):
        selected = metadata_regis.select_unit_contains(substring)
        assert isinstance(selected, RegisUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert_array_equal(
            selected.unit,
            ["NUDNz", "NUDNz1", "NUOOz", "NUOOz1", "NUOOk1", "NUOOz2", "NUOOc"],
        )

    @pytest.mark.unittest
    def test_select_unit_contains_error(self, metadata_regis):
        non_existing_substring = "NONEXISTENT_SUBSTRING"
        with pytest.raises(MissingUnitError):
            metadata_regis.select_unit_contains(non_existing_substring)

    @pytest.mark.unittest
    def test_select_description_contains_single_value(self, metadata_regis):
        selected = metadata_regis.select_description_contains("Formatie van Oosterhout")
        assert isinstance(selected, RegisUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert_array_equal(
            selected.unit,
            ["NUOOz", "NUOOz1", "NUOOk1", "NUOOz2", "NUOOc"],
        )

    @pytest.mark.parametrize(
        "substring",
        [
            ["Formatie van Oosterhout", "Formatie van Drachten"],
            np.array(["Formatie van Oosterhout", "Formatie van Drachten"]),
            pd.Series(["Formatie van Oosterhout", "Formatie van Drachten"]),
        ],
        ids=["list-of-values", "numpy-array", "pandas-series"],
    )
    def test_select_description_contains_list_of_values(
        self, metadata_regis, substring
    ):
        selected = metadata_regis.select_description_contains(substring)
        assert isinstance(selected, RegisUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert_array_equal(
            selected.unit,
            ["NUDNz", "NUDNz1", "NUOOz", "NUOOz1", "NUOOk1", "NUOOz2", "NUOOc"],
        )

    @pytest.mark.unittest
    def test_select_description_contains_error(self, metadata_regis):
        non_existing_substring = "NONEXISTENT_SUBSTRING"
        with pytest.raises(MissingUnitError):
            metadata_regis.select_description_contains(non_existing_substring)
