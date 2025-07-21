from typing import Literal

import pytest
from numpy.testing import assert_array_equal

from geost.bro import GeoTop
from geost.bro.bro_geotop import StratGeotop


class TestGeoTop:
    @pytest.fixture
    def test_area(self):
        """
        xmin, ymin, xmax, ymax bounding box of a test area.
        """
        return (115_000, 500_000, 115_500, 500_500)

    @pytest.mark.unittest
    def test_lazy_load_from_opendap(self):
        geotop = GeoTop.from_opendap(lazy=True)
        assert geotop.ds.chunks == {
            "y": (75, 684, 684, 684, 684),
            "x": (643, 643, 643, 643, 74),
            "z": (76, 76, 76, 76, 9),
        }

    @pytest.mark.unittest
    def test_from_opendap(
        self,
        test_area: tuple[
            Literal[115000], Literal[500000], Literal[115500], Literal[500500]
        ],
    ):
        geotop = GeoTop.from_opendap(bbox=test_area)
        assert isinstance(geotop, GeoTop)
        assert geotop.resolution == (100, 100, 0.5)
        assert geotop["strat"].dims == ("y", "x", "z")
        assert geotop.crs == 28992
        assert_array_equal(
            geotop["x"], [114_950, 115_050, 115_150, 115_250, 115_350, 115_450, 115_550]
        )
        assert_array_equal(
            geotop["y"], [500_550, 500_450, 500_350, 500_250, 500_150, 500_050, 499_950]
        )
        assert_array_equal(geotop["z"][:3], [-49.75, -49.25, -48.75])


class TestStratGeotop:
    @pytest.mark.unittest
    def test_select_units(self):
        sel = StratGeotop.select_units(["AAOP", "OEC", "BXDE", "ANAWA"])
        assert_array_equal(
            sel,
            [
                StratGeotop.holocene.OEC,
                StratGeotop.channel.ANAWA,
                StratGeotop.older.BXDE,
                StratGeotop.antropogenic.AAOP,
            ],
        )
        assert_array_equal(sel, [1070, 6010, 3040, 1000])

        sel = StratGeotop.select_units("AEC")
        assert_array_equal(sel, [StratGeotop.channel.AEC])
        assert_array_equal(sel, [6000])

    @pytest.mark.unittest
    def test_select_values(self):
        sel = StratGeotop.select_values([1070, 6010, 3040, 1000])
        assert_array_equal(
            sel,
            [
                StratGeotop.holocene.OEC,
                StratGeotop.channel.ANAWA,
                StratGeotop.older.BXDE,
                StratGeotop.antropogenic.AAOP,
            ],
        )
        assert_array_equal(sel, [1070, 6010, 3040, 1000])

        sel = StratGeotop.select_values(6000)
        assert_array_equal(sel, [StratGeotop.channel.AEC])
        assert_array_equal(sel, [6000])

    @pytest.mark.unittest
    def test_select_units_empty(self):
        sel = StratGeotop.select_units("foo")
        assert not sel

    @pytest.mark.unittest
    def test_select_values_empty(self):
        sel = StratGeotop.select_values(-9999)
        assert not sel

    @pytest.mark.unittest
    def test_to_dict(self):
        d = StratGeotop.to_dict(key="unit")
        assert d == {
            "NIGR": 1010,
            "NINB": 1045,
            "NASC": 1020,
            "ONAWA": 1030,
            "NAZA": 1040,
            "NAWAZU": 1048,
            "NAWAAL": 1049,
            "NAWA": 1050,
            "BHEC": 1060,
            "OEC": 1070,
            "NAWOBE": 1080,
            "KK1": 1085,
            "NIFL": 1089,
            "NIHO": 1090,
            "NAZA2": 1095,
            "NAWO": 1100,
            "NWNZ": 1110,
            "NAWOVE": 1120,
            "KK2": 1125,
            "NIBA": 1130,
            "NA": 2000,
            "EC": 2010,
            "NI": 2020,
            "KK": 2030,
            "AEC": 6000,
            "ABEOM": 6005,
            "ANAWA": 6010,
            "ANAWO": 6020,
            "BEC": 6100,
            "BNAWA": 6110,
            "BNAWO": 6120,
            "CEC": 6200,
            "CNAWA": 6210,
            "CNAWO": 6220,
            "DEC": 6300,
            "DNAWA": 6310,
            "DNAWO": 6320,
            "EEC": 6400,
            "ENAWA": 6410,
            "ENAWO": 6420,
            "BXKO": 3000,
            "BXSI": 3010,
            "BXSI1": 3011,
            "BXWI": 3020,
            "BXSI2": 3012,
            "BXWIKO": 3025,
            "BXWISIKO": 3030,
            "BXDE": 3040,
            "BXDEKO": 3045,
            "BXSC": 3050,
            "BXLM": 3060,
            "BXBS": 3090,
            "BX": 3100,
            "KRWY": 4000,
            "KRBXDE": 4010,
            "KRZU": 4020,
            "KROE": 4030,
            "KRTW": 4040,
            "KR": 4050,
            "BEOM": 4055,
            "BEWY": 4060,
            "BERO": 4070,
            "BE": 4080,
            "KW1": 4085,
            "KW": 4090,
            "WB": 4100,
            "EE": 4110,
            "EEWB": 4120,
            "DR": 5000,
            "DRGI": 5010,
            "GE": 5020,
            "DN": 5030,
            "URTY": 5040,
            "PE": 5050,
            "UR": 5060,
            "ST": 5070,
            "AP": 5080,
            "SY": 5090,
            "PZ": 5100,
            "WA": 5110,
            "PZWA": 5120,
            "MS": 5130,
            "KI": 5140,
            "OO": 5150,
            "IE": 5160,
            "VI": 5170,
            "BR": 5180,
            "VE": 5185,
            "RUBO": 5190,
            "RU": 5200,
            "TOZEWA": 5210,
            "TOGO": 5220,
            "TO": 5230,
            "DOAS": 5240,
            "DOIE": 5250,
            "DO": 5260,
            "LA": 5270,
            "HT": 5280,
            "HO": 5290,
            "MT": 5300,
            "GU": 5310,
            "VA": 5320,
            "AK": 5330,
            "AAOP": 1000,
            "AAES": 1005,
        }
        assert len(d) == (
            len(StratGeotop.holocene)
            + len(StratGeotop.channel)
            + len(StratGeotop.older)
            + len(StratGeotop.antropogenic)
        )
