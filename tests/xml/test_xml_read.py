from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.io import xml
from geost.io.xml import schemas


@pytest.fixture
def xml_string(testdatadir: Path):
    with open(testdatadir / r"xml/bhrgt_bro.xml", "rb") as file:
        string = file.read()
    return string


@pytest.mark.parametrize(
    "file, reader, expected_end_nap",
    [
        ("bhrgt_bro.xml", xml.read_bhrgt, -6.91),
        ("bhrg_bro.xml", xml.read_bhrg, -2.31),
        ("bhrp_bro.xml", xml.read_bhrp, -1.29),
        ("cpt_bro.xml", xml.read_cpt, -6.48),
        ("sfr_bro.xml", xml.read_sfr, -0.73),
    ],
    ids=["bhrgt", "bhrg", "bhrp", "cpt", "sfr"],
)
def test_read(testdatadir: Path, file: str, reader: Callable, expected_end_nap: float):
    bro_xml = testdatadir / r"xml" / file
    header, data = xml.read(bro_xml, reader)
    assert isinstance(header, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)
    expected_columns_present = ["nr", "x", "y", "surface", "end"]
    assert all(c in data.columns for c in expected_columns_present)

    if "top" in data.columns and "bottom" in data.columns:
        assert data["top"].dtype == data["bottom"].dtype == float

    assert_array_almost_equal(header["end"], expected_end_nap)
    assert_array_almost_equal(data["end"], expected_end_nap)


class TestBhrgt:
    @pytest.fixture
    def custom_bhrgt_schema(self):
        return {
            "payload_root": "dispatchDocument",
            "bro_id": {"xpath": "brocom:broId"},
            "date": {"xpath": "reportHistory/reportStartDate/brocom:date"},
        }

    @pytest.mark.unittest
    def test_read_bhrgt(self, testdatadir: Path):
        bro_xml = testdatadir / r"xml/bhrgt_bro.xml"

        data = xml.read_bhrgt(bro_xml)

        assert isinstance(data, dict)
        assert data["nr"] == "BHR000000336600"

        with pytest.raises(SyntaxError, match="Invalid xml schema"):
            xml.read_bhrgt(bro_xml, company="Wiertsema")

        with pytest.raises(
            ValueError, match="No predefined schema for 'UnknownCompany'"
        ):
            xml.read_bhrgt(bro_xml, company="UnknownCompany")

    @pytest.mark.unittest
    def test_read_bhrgt_with_custom_schema(
        self, testdatadir: Path, custom_bhrgt_schema: dict[str, Any]
    ):
        data = xml.read_bhrgt(
            testdatadir / r"xml/bhrgt_bro.xml", schema=custom_bhrgt_schema
        )
        assert isinstance(data, dict)
        assert data == {"bro_id": "BHR000000336600", "date": "2020-04-15"}

    @pytest.mark.unittest
    def test_read_bhrgt_from_string(self, xml_string: bytes):
        data = xml.read_bhrgt(xml_string)
        assert isinstance(data, dict)
        assert data["nr"] == "BHR000000336600"

    @pytest.mark.unittest
    def test_read_bhrgt_with_custom_schema_no_payload_root(self, testdatadir: Path):
        """
        Test reading an xml string without a payload root. This can lead to unexpected results
        as the XML root can contain multiple elements with additional information besides the
        main data element.

        """
        schema = {"bro_id": {"xpath": "BHR_GT_O/brocom:broId"}}
        with pytest.warns(UserWarning, match="Multiple payloads found in XML"):
            data = xml.read_bhrgt(
                testdatadir / r"xml/bhrgt_bro.xml", schema=schema, read_all=False
            )
            assert isinstance(data, dict)

        data = xml.read_bhrgt(
            testdatadir / r"xml/bhrgt_bro.xml", schema=schema, read_all=True
        )
        assert isinstance(data, list)
        assert data[0] == {"bro_id": None}
        assert data[1] == {"bro_id": None}
        assert data[2] == {"bro_id": None}
        assert data[3] == {"bro_id": "BHR000000336600"}

    @pytest.mark.unittest
    def test_read_bhrgt_bro(self, testdatadir: Path):
        data = xml.read_bhrgt(testdatadir / r"xml/bhrgt_bro.xml")

        assert isinstance(data, dict)
        assert data["nr"] == "BHR000000336600"
        assert data["location"] == (132781.327, 448031.1)
        assert data["crs"] == "urn:ogc:def:crs:EPSG::28992"
        assert data["surface"] == 0.09
        assert data["vertical_datum"] == "NAP"
        assert data["groundwater_level"] == 1.6
        assert data["end"] == 7.0
        assert data["data"] == {
            "upperBoundary": [
                "0.00",
                "1.00",
                "1.70",
                "2.00",
                "2.40",
                "3.40",
                "4.20",
                "5.00",
                "6.00",
            ],
            "lowerBoundary": [
                "1.00",
                "1.70",
                "2.00",
                "2.40",
                "3.40",
                "4.20",
                "5.00",
                "6.00",
                "7.00",
            ],
            "geotechnicalSoilName": [
                "sterkZandigeKleiMetGrind",
                "zwakGrindigZand",
                "zwakZandigeKleiMetGrind",
                "zand",
                "detritus",
                "detritus",
                "zwakZandigeKlei",
                "zand",
                "zand",
            ],
        }

    @pytest.mark.unittest
    def test_read_bhrgt_wiertsema(self, testdatadir: Path):
        data = xml.read_bhrgt(
            testdatadir / r"xml/bhrgt_wiertsema.xml", company="Wiertsema"
        )

        assert isinstance(data, dict)
        assert data["nr"] == "_87078_HB001"
        assert data["location"] == (182243.9, 335073.8)
        assert data["crs"] == "urn:ogc:def:crs:EPSG::28992"
        assert data["surface"] == 37.74
        assert data["vertical_datum"] == "NAP"
        assert data["groundwater_level"] == 1.1
        assert data["end"] == 2.2
        assert data["data"] == {
            "upperBoundary": ["0.00", "0.10", "0.40", "0.70", "1.70"],
            "lowerBoundary": ["0.10", "0.40", "0.70", "1.70", "2.20"],
            "geotechnicalSoilName": [
                "zwakZandigSilt",
                "zwakZandigGrind",
                "klei",
                "zand",
                "zand",
            ],
        }


class TestBhrg:
    @pytest.mark.unittest
    def test_read_bhrg(self, testdatadir: Path):
        bro_xml = testdatadir / r"xml/bhrg_bro.xml"

        data = xml.read_bhrg(bro_xml)
        assert isinstance(data, dict)

        with pytest.raises(
            ValueError, match="No predefined schema for 'UnknownCompany'"
        ):
            xml.read_bhrg(bro_xml, company="UnknownCompany")

        invalid_schema = schemas.bhrgt.get("Wiertsema", None)
        with pytest.raises(SyntaxError, match="Invalid xml schema"):
            xml.read_bhrg(bro_xml, schema=invalid_schema)

    @pytest.mark.unittest
    def test_read_bhrg_bro(self, testdatadir: Path):
        data = xml.read_bhrg(testdatadir / r"xml/bhrg_bro.xml")

        assert isinstance(data, dict)
        assert data["nr"] == "BHR000000396406"
        assert data["location"] == (126149.0, 452162.0)
        assert data["crs"] == "urn:ogc:def:crs:EPSG::28992"
        assert data["surface"] == 0.69
        assert data["vertical_datum"] == "NAP"
        assert data["end"] == 3.0
        assert data["data"] == {
            "upperBoundary": ["0.000", "0.250", "1.600", "2.000", "2.500"],
            "lowerBoundary": ["0.250", "1.600", "2.000", "2.500", "3.000"],
            "soilNameNEN5104": [
                "zwakZandigeKlei",
                "sterkSiltigeKlei",
                "zwakZandigeKlei",
                "sterkZandigeKlei",
                "zwakSiltigZand",
            ],
        }


class TestCpt:
    @pytest.mark.unittest
    def test_read_cpt(self, testdatadir: Path):
        data = xml.read_cpt(testdatadir / r"xml/cpt_bro.xml")

        assert isinstance(data, dict)

        with pytest.raises(
            ValueError, match="No predefined schema for 'UnknownCompany'"
        ):
            xml.read_cpt(testdatadir / r"xml/cpt_bro.xml", company="UnknownCompany")

        invalid_schema = schemas.bhrgt.get("Wiertsema", None)
        with pytest.raises(SyntaxError, match="Invalid xml schema"):
            xml.read_cpt(testdatadir / r"xml/cpt_bro.xml", schema=invalid_schema)

    @pytest.mark.unittest
    def test_read_cpt_bro(self, testdatadir: Path):
        data = xml.read_cpt(testdatadir / r"xml/cpt_bro.xml")

        assert isinstance(data, dict)
        assert data["nr"] == "CPT000000155283"
        assert data["location"] == (132782.52, 448030.34)
        assert data["crs"] == "urn:ogc:def:crs:EPSG::28992"
        assert data["surface"] == 0.09
        assert data["vertical_datum"] == "NAP"
        assert data["predrilled_depth"] == 0.5
        assert data["end"] == 6.57
        assert_array_equal(
            list(data["data"].keys()),
            [
                "penetrationlength",
                "depth",
                "elapsedtime",
                "coneresistance",
                "correctedconeresistance",
                "netconeresistance",
                "magneticfieldstrengthx",
                "magneticfieldstrengthy",
                "magneticfieldstrengthz",
                "magneticfieldstrengthtotal",
                "electricalconductivity",
                "inclinationew",
                "inclinationns",
                "inclinationx",
                "inclinationy",
                "inclinationresultant",
                "magneticinclination",
                "magneticdeclination",
                "localfriction",
                "poreratio",
                "temperature",
                "porepressureu1",
                "porepressureu2",
                "porepressureu3",
                "frictionratio",
            ],
        )
        original_missing_value = -999999
        processed_data = np.array(list(data["data"].values()))
        assert not np.any(processed_data == original_missing_value)


class TestBhrp:
    @pytest.mark.unittest
    def test_read_bhrp(self, testdatadir: Path):
        data = xml.read_bhrp(testdatadir / r"xml/bhrp_bro.xml")
        assert isinstance(data, dict)

        with pytest.raises(
            ValueError, match="No predefined schema for 'UnknownCompany'"
        ):
            xml.read_bhrp(testdatadir / r"xml/bhrp_bro.xml", company="UnknownCompany")

        invalid_schema = schemas.bhrgt.get("Wiertsema", None)
        with pytest.raises(SyntaxError, match="Invalid xml schema"):
            xml.read_bhrp(testdatadir / r"xml/bhrp_bro.xml", schema=invalid_schema)

    @pytest.mark.unittest
    def test_read_bhrp_bro(self, testdatadir: Path):
        data = xml.read_bhrp(testdatadir / r"xml/bhrp_bro.xml")

        assert isinstance(data, dict)
        assert data["nr"] == "BHR000000108193"
        assert data["location"] == (129491.0, 452255.0)
        assert data["crs"] == "urn:ogc:def:crs:EPSG::28992"
        assert data["surface"] == 0.21
        assert data["vertical_datum"] == "NAP"
        assert data["begin_depth"] == 0.0
        assert data["end"] == 1.5
        assert data["ghg"] == 0.55
        assert data["glg"] == 1.6
        assert data["landuse"] == "graslandBlijvend"
        assert data["data"] == {
            "upperBoundary": ["0.000", "0.600", "0.900", "1.200", "1.300", "1.400"],
            "lowerBoundary": ["0.600", "0.900", "1.200", "1.300", "1.400", "1.500"],
            "standardSoilName": [
                "sterkSiltigeKlei",
                "sterkSiltigeKlei",
                "sterkSiltigeKlei",
                "uiterstSiltigeKlei",
                "zwakSiltigZand",
                "zwakSiltigZand",
            ],
        }


class TestSfr:
    @pytest.mark.unittest
    def test_read_sfr(self, testdatadir: Path):
        data = xml.read_sfr(testdatadir / r"xml/sfr_bro.xml")
        assert isinstance(data, dict)

        with pytest.raises(
            ValueError, match="No predefined schema for 'UnknownCompany'"
        ):
            xml.read_sfr(testdatadir / r"xml/sfr_bro.xml", company="UnknownCompany")

        invalid_schema = schemas.bhrgt.get("Wiertsema", None)
        with pytest.raises(SyntaxError, match="Invalid xml schema"):
            xml.read_sfr(testdatadir / r"xml/sfr_bro.xml", schema=invalid_schema)

    @pytest.mark.unittest
    def test_read_sfr_bro(self, testdatadir: Path):
        data = xml.read_sfr(testdatadir / r"xml/sfr_bro.xml")

        assert isinstance(data, dict)
        assert data["nr"] == "SFR000000000687"
        assert data["location"] == (132250.0, 451075.0)
        assert data["crs"] == "urn:ogc:def:crs:EPSG::28992"
        assert data["surface"] == 0.87
        assert data["vertical_datum"] == "NAP"
        assert data["landuse"] == "grasland"
        assert data["outcrop_type"] == "profielkuil"
        assert data["end"] == 1.6
        assert data["data"] == {
            "upperBoundary": [
                "0.000",
                "0.070",
                "0.250",
                "0.550",
                "0.830",
                "1.200",
                "1.400",
            ],
            "lowerBoundary": [
                "0.070",
                "0.250",
                "0.550",
                "0.830",
                "1.200",
                "1.400",
                "1.600",
            ],
            "soilNameNEN5104": [
                "sterkSiltigeKlei",
                "sterkSiltigeKlei",
                "matigSiltigeKlei",
                "sterkSiltigeKlei",
                "uiterstSiltigeKlei",
                "matigZandigeKlei",
                "sterkZandigeKlei",
            ],
        }
