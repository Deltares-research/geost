import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost import read_geotop_netcdf
from geost.bro.geotop import (
    GeotopUnits,
    UnitType,
    geotop_lithok_units,
    geotop_strat_units,
)
from geost.exceptions import MissingUnitError


@pytest.mark.xfail(
    reason="Will fail in CI because pooch data can only be found from main branch"
)
@pytest.mark.unittest
def test_geotop_strat_units():
    metadata = geotop_strat_units()
    assert isinstance(metadata, GeotopUnits)
    assert isinstance(metadata.df, pd.DataFrame)
    assert metadata.unit_type == UnitType.STRAT
    assert metadata.data_var == "strat"
    assert isinstance(metadata.voxel_nr, pd.Index)
    assert isinstance(metadata.unit, pd.Series)
    assert isinstance(metadata.description, pd.Series)
    assert isinstance(metadata.colors_rgb, pd.DataFrame)
    assert isinstance(metadata.geotop_version, str)
    assert metadata.geotop_version == "v01r6s1"


@pytest.mark.xfail(
    reason="Will fail in CI because pooch data can only be found from main branch"
)
@pytest.mark.unittest
def test_geotop_lithok_units():
    metadata = geotop_lithok_units()
    assert isinstance(metadata, GeotopUnits)
    assert isinstance(metadata.df, pd.DataFrame)
    assert metadata.unit_type == UnitType.LITHOK
    assert isinstance(metadata.voxel_nr, pd.Index)
    assert metadata.data_var == "lithok"
    assert isinstance(metadata.unit, pd.Series)
    assert isinstance(metadata.description, pd.Series)
    assert isinstance(metadata.colors_rgb, pd.DataFrame)
    assert isinstance(metadata.geotop_version, str)
    assert metadata.geotop_version == "v01r6s1"


class TestGeotopMetadata:
    @pytest.mark.unittest
    def test_voxel_nr_property(self, metadata_strat):
        assert isinstance(metadata_strat.voxel_nr, pd.Index)
        assert metadata_strat.voxel_nr.name == "VOXEL_NR"

    @pytest.mark.unittest
    def test_unit_property(self, metadata_strat):
        assert isinstance(metadata_strat.unit, pd.Series)
        assert metadata_strat.unit.name == "STR_UNIT_CD"
        assert_array_equal(metadata_strat.unit.index, metadata_strat.voxel_nr)

    @pytest.mark.unittest
    def test_description_property(self, metadata_strat):
        assert isinstance(metadata_strat.description, pd.Series)
        assert metadata_strat.description.name == "DESCRIPTION"
        assert_array_equal(metadata_strat.description.index, metadata_strat.voxel_nr)

    @pytest.mark.unittest
    def test_colors_rgb(self, metadata_strat):
        assert isinstance(metadata_strat.colors_rgb, pd.DataFrame)
        assert_array_equal(metadata_strat.colors_rgb.index, metadata_strat.voxel_nr)
        assert_array_equal(
            metadata_strat.colors_rgb.columns, ["RED_DEC", "GREEN_DEC", "BLUE_DEC"]
        )

    @pytest.mark.unittest
    def test_geotop_version(self, metadata_strat):
        assert isinstance(metadata_strat.geotop_version, str)
        assert metadata_strat.geotop_version == "v01r6s1"

    @pytest.mark.unittest
    def test_check_version(self, metadata_strat, geotop_small):
        assert metadata_strat.check_version_matches(geotop_small) is True

        with pytest.raises(
            TypeError,
            match="The model version cannot be found in a DataArray of a GeoTOP variable",
        ):
            metadata_strat.check_version_matches(geotop_small["strat"])

        with pytest.warns(UserWarning, match="GeoTOP version mismatch"):
            geotop_small.attrs["title"] = "v01r5s1"
            assert metadata_strat.check_version_matches(geotop_small) is False

    @pytest.mark.parametrize(
        "values", [1130, [1130, 2010]], ids=["single-value", "list-of-values"]
    )
    def test_select_voxel_nr(self, metadata_strat, values):
        selected = metadata_strat.select_voxel_nr(values)
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(selected.voxel_nr, values)

    @pytest.mark.unittest
    def test_select_voxel_nr_with_geotop(self, metadata_strat, geotop_small):
        selected = metadata_strat.select_voxel_nr(geotop_small["strat"])
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(
            selected.voxel_nr,
            [1070, 1090, 1130, 2010, 4000, 4010, 5060, 5070, 5120, 6400],
        )

        # Should work on the entire dataset as well, not just a single variable
        selected = metadata_strat.select_voxel_nr(geotop_small)
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(
            selected.voxel_nr,
            [1070, 1090, 1130, 2010, 4000, 4010, 5060, 5070, 5120, 6400],
        )

    @pytest.mark.unittest
    def test_select_voxel_nr_error(self, metadata_strat):
        non_existing_voxel_nr = 9999
        with pytest.raises(MissingUnitError):
            metadata_strat.select_voxel_nr(non_existing_voxel_nr)

    @pytest.mark.parametrize(
        "units",
        ["NUNIHO", ["NUNIHO", "NUNIBA"]],
        ids=["single-value", "list-of-values"],
    )
    def test_select_unit(self, metadata_strat, units):
        selected = metadata_strat.select_unit(units)
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(selected.unit, units)

    @pytest.mark.unittest
    def test_select_unit_error(self, metadata_strat):
        non_existing_unit = "NONEXISTENT_UNIT"
        with pytest.raises(MissingUnitError):
            metadata_strat.select_unit(non_existing_unit)

    @pytest.mark.unittest
    def test_select_unit_contains_single_value(self, metadata_strat):
        selected = metadata_strat.select_unit_contains("NAWA")
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(selected.voxel_nr, [1030, 1048, 1049, 1050, 6010, 6110])
        assert_array_equal(
            selected.unit,
            ["NUNAWA1", "NUNAWAZU", "NUNAWAAL", "NUNAWA2", "NUNAWAga", "NUNAWAgb"],
        )

    @pytest.mark.parametrize(
        "substring",
        [["NIHO", "NIBA"], np.array(["NIHO", "NIBA"]), pd.Series(["NIHO", "NIBA"])],
        ids=["list-of-values", "numpy-array", "pandas-series"],
    )
    def test_select_unit_contains_list_of_values(self, metadata_strat, substring):
        selected = metadata_strat.select_unit_contains(substring)
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(selected.voxel_nr, [1090, 1130])
        assert_array_equal(selected.unit, ["NUNIHO", "NUNIBA"])

    @pytest.mark.unittest
    def test_select_unit_contains_case_insensitive(self, metadata_strat):
        selected = metadata_strat.select_unit_contains("niho", case_sensitive=False)
        assert isinstance(selected, GeotopUnits)
        assert_array_equal(selected.voxel_nr, 1090)
        assert_array_equal(selected.unit, "NUNIHO")

        # Mixed case with case_sensitive=True should only return the exact match
        selected = metadata_strat.select_unit_contains(
            ["niho", "NIBA"], case_sensitive=True
        )
        assert isinstance(selected, GeotopUnits)
        assert_array_equal(selected.voxel_nr, 1130)
        assert_array_equal(selected.unit, "NUNIBA")

        with pytest.raises(MissingUnitError):
            metadata_strat.select_unit_contains("niho", case_sensitive=True)

    @pytest.mark.unittest
    def test_select_unit_contains_error(self, metadata_strat):
        non_existing_substring = "NONEXISTENT_SUBSTRING"
        with pytest.raises(MissingUnitError):
            metadata_strat.select_unit_contains(non_existing_substring)

    @pytest.mark.unittest
    def test_select_description_contains_single_value(self, metadata_strat):
        selected = metadata_strat.select_description_contains("Formatie van Naaldwijk")
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(
            selected.voxel_nr,
            [
                1020,
                1030,
                1040,
                1048,
                1049,
                1050,
                1080,
                1095,
                1100,
                1120,
                2000,
                6010,
                6110,
                6320,
                6420,
            ],
        )
        assert_array_equal(
            selected.unit,
            [
                "NUNASC",
                "NUNAWA1",
                "NUNAZA1",
                "NUNAWAZU",
                "NUNAWAAL",
                "NUNAWA2",
                "NUNAWOBE",
                "NUNAZA2",
                "NUNAWO",
                "NUNAWOVE",
                "NUNA",
                "NUNAWAga",
                "NUNAWAgb",
                "NUNAWOgd",
                "NUNAWOge",
            ],
        )

    @pytest.mark.parametrize(
        "substring",
        [
            ["Formatie van Nieuwkoop", "Formatie van Naaldwijk"],
            np.array(["Formatie van Nieuwkoop", "Formatie van Naaldwijk"]),
            pd.Series(["Formatie van Nieuwkoop", "Formatie van Naaldwijk"]),
        ],
        ids=["list-of-values", "numpy-array", "pandas-series"],
    )
    def test_select_description_contains_list_of_values(
        self, metadata_strat, substring
    ):
        selected = metadata_strat.select_description_contains(substring)
        assert isinstance(selected, GeotopUnits)
        assert isinstance(selected.df, pd.DataFrame)
        assert selected.unit_type == UnitType.STRAT
        assert_array_equal(
            selected.voxel_nr,
            [
                1010,
                1045,
                1020,
                1030,
                1040,
                1048,
                1049,
                1050,
                1070,
                1080,
                1085,
                1089,
                1090,
                1095,
                1100,
                1120,
                1125,
                1130,
                2000,
                2010,
                2020,
                6010,
                6110,
                6320,
                6420,
            ],
        )
        assert_array_equal(
            selected.unit,
            [
                "NUNIGR",
                "NUNInb",
                "NUNASC",
                "NUNAWA1",
                "NUNAZA1",
                "NUNAWAZU",
                "NUNAWAAL",
                "NUNAWA2",
                "NUEC1",
                "NUNAWOBE",
                "NUKK1",
                "NUNIFL",
                "NUNIHO",
                "NUNAZA2",
                "NUNAWO",
                "NUNAWOVE",
                "NUKK2",
                "NUNIBA",
                "NUNA",
                "NUEC2",
                "NUNI",
                "NUNAWAga",
                "NUNAWAgb",
                "NUNAWOgd",
                "NUNAWOge",
            ],
        )

    @pytest.mark.unittest
    def test_select_description_contains_case_insensitive(self, metadata_strat):
        selected = metadata_strat.select_description_contains(
            "formatie van nieuwkoop", case_sensitive=False
        )
        assert isinstance(selected, GeotopUnits)
        assert_array_equal(
            selected.voxel_nr,
            [1010, 1045, 1040, 1070, 1085, 1089, 1090, 1095, 1125, 1130, 2010, 2020],
        )

        # Mixed case with case_sensitive=True should only return the exact match
        selected = metadata_strat.select_description_contains(
            ["Formatie van Nieuwkoop", "formatie van naaldwijk"], case_sensitive=True
        )
        assert isinstance(selected, GeotopUnits)
        assert_array_equal(
            selected.voxel_nr,
            [1010, 1045, 1040, 1070, 1085, 1089, 1090, 1095, 1125, 1130, 2010, 2020],
        )

        with pytest.raises(MissingUnitError):
            metadata_strat.select_description_contains(
                "formatie van nieuwkoop", case_sensitive=True
            )

    @pytest.mark.unittest
    def test_select_description_contains_error(self, metadata_strat):
        non_existing_substring = "NONEXISTENT_SUBSTRING"
        with pytest.raises(MissingUnitError):
            metadata_strat.select_description_contains(non_existing_substring)

    @pytest.mark.unittest
    def test_get_antropogenic_units(self, metadata_strat, metadata_lithok):
        with pytest.raises(
            ValueError,
            match="Stratigraphic units are not present in the lithologic metadata of GeoTOP.",
        ):
            metadata_lithok.get_antropogenic_units()

        antropogenic_units = metadata_strat.get_antropogenic_units()
        assert isinstance(antropogenic_units, GeotopUnits)
        assert isinstance(antropogenic_units.df, pd.DataFrame)
        assert antropogenic_units.unit_type == UnitType.STRAT
        assert_array_equal(antropogenic_units.voxel_nr, [1000, 1005])
        assert_array_equal(antropogenic_units.unit, ["NUAAOP", "NUAAES"])

    @pytest.mark.unittest
    def test_get_holocene_channel_units(self, metadata_strat, metadata_lithok):
        with pytest.raises(
            ValueError,
            match="Stratigraphic units are not present in the lithologic metadata of GeoTOP.",
        ):
            metadata_lithok.get_holocene_channel_units()

        holocene_channel_units = metadata_strat.get_holocene_channel_units()
        assert isinstance(holocene_channel_units, GeotopUnits)
        assert isinstance(holocene_channel_units.df, pd.DataFrame)
        assert holocene_channel_units.unit_type == UnitType.STRAT
        assert_array_equal(
            holocene_channel_units.voxel_nr,
            [6000, 6005, 6010, 6100, 6110, 6200, 6300, 6320, 6400, 6420],
        )
        assert_array_equal(
            holocene_channel_units.unit,
            [
                "NUECga",
                "NUBEOMga",
                "NUNAWAga",
                "NUECgb",
                "NUNAWAgb",
                "NUECgc",
                "NUECgd",
                "NUNAWOgd",
                "NUECge",
                "NUNAWOge",
            ],
        )

    @pytest.mark.unittest
    def test_get_holocene_units(self, metadata_strat, metadata_lithok):
        with pytest.raises(
            ValueError,
            match="Stratigraphic units are not present in the lithologic metadata of GeoTOP.",
        ):
            metadata_lithok.get_holocene_units()

        holocene = metadata_strat.get_holocene_units()
        assert isinstance(holocene, GeotopUnits)
        assert isinstance(holocene.df, pd.DataFrame)
        assert holocene.unit_type == UnitType.STRAT
        assert_array_equal(
            holocene.voxel_nr,
            [
                1010,
                1045,
                1020,
                1030,
                1040,
                1048,
                1049,
                1050,
                1070,
                1080,
                1085,
                1089,
                1090,
                1095,
                1100,
                1120,
                1125,
                1130,
                2000,
                2010,
                2020,
                6000,
                6005,
                6010,
                6100,
                6110,
                6200,
                6300,
                6320,
                6400,
                6420,
            ],
        )
        assert_array_equal(
            holocene.unit,
            [
                "NUNIGR",
                "NUNInb",
                "NUNASC",
                "NUNAWA1",
                "NUNAZA1",
                "NUNAWAZU",
                "NUNAWAAL",
                "NUNAWA2",
                "NUEC1",
                "NUNAWOBE",
                "NUKK1",
                "NUNIFL",
                "NUNIHO",
                "NUNAZA2",
                "NUNAWO",
                "NUNAWOVE",
                "NUKK2",
                "NUNIBA",
                "NUNA",
                "NUEC2",
                "NUNI",
                "NUECga",
                "NUBEOMga",
                "NUNAWAga",
                "NUECgb",
                "NUNAWAgb",
                "NUECgc",
                "NUECgd",
                "NUNAWOgd",
                "NUECge",
                "NUNAWOge",
            ],
        )

        holocene = metadata_strat.get_holocene_units(include_channel_units=False)
        assert isinstance(holocene, GeotopUnits)
        assert isinstance(holocene.df, pd.DataFrame)
        assert holocene.unit_type == UnitType.STRAT
        assert_array_equal(
            holocene.voxel_nr,
            [
                1010,
                1045,
                1020,
                1030,
                1040,
                1048,
                1049,
                1050,
                1070,
                1080,
                1085,
                1089,
                1090,
                1095,
                1100,
                1120,
                1125,
                1130,
                2000,
                2010,
                2020,
            ],
        )
        assert_array_equal(
            holocene.unit,
            [
                "NUNIGR",
                "NUNInb",
                "NUNASC",
                "NUNAWA1",
                "NUNAZA1",
                "NUNAWAZU",
                "NUNAWAAL",
                "NUNAWA2",
                "NUEC1",
                "NUNAWOBE",
                "NUKK1",
                "NUNIFL",
                "NUNIHO",
                "NUNAZA2",
                "NUNAWO",
                "NUNAWOVE",
                "NUKK2",
                "NUNIBA",
                "NUNA",
                "NUEC2",
                "NUNI",
            ],
        )
