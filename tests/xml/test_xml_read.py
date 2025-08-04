from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost.io import xml


@pytest.fixture
def xml_string(testdatadir: Path):
    with open(testdatadir / r"xml/bhrgt_bro.xml", "rb") as file:
        string = file.read()
    return string


@pytest.mark.parametrize(
    "file, reader, expected",
    [
        ("bhrgt_bro.xml", xml.read_bhrgt, pd.DataFrame),
        ("bhrg_bro.xml", xml.read_bhrg, pytest.raises(NotImplementedError)),
        ("bhrp_bro.xml", xml.read_bhrp, pytest.raises(NotImplementedError)),
        ("cpt_bro.xml", xml.read_cpt, pytest.raises(NotImplementedError)),
        ("sfr_bro.xml", xml.read_sfr, pytest.raises(NotImplementedError)),
    ],
)
def test_read(testdatadir: Path, file: str, reader: Callable, expected: Any):
    bro_xml = testdatadir / r"xml" / file

    if isinstance(expected, type(pytest.raises(Exception))):
        with expected:
            xml.read(bro_xml, reader)
    else:
        header, data = xml.read(bro_xml, reader)
        assert isinstance(header, expected)
        assert isinstance(data, expected)
        expected_columns_present = ["nr", "x", "y", "surface", "end", "top", "bottom"]
        assert all(c in data.columns for c in expected_columns_present)


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
    pass


class TestCpt:
    pass


class TestBhrp:
    pass
