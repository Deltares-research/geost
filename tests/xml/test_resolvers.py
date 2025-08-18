from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lxml import etree
from numpy.testing import assert_array_equal

from geost.io.xml import resolvers


def crs_element():
    el = etree.Element("Point", srsName="urn:ogc:def:crs:EPSG::28992")
    return el


def empty_crs_element():
    el = etree.Element("Point")
    return el


@pytest.fixture
def bhrgt_no_namespace(testdatadir: Path):
    root = etree.parse(testdatadir / r"xml/bhrgt_data_element_no_namespaces.xml")
    return root


@pytest.mark.parametrize(
    "element, expected",
    [
        (crs_element(), "urn:ogc:def:crs:EPSG::28992"),
        (empty_crs_element(), "unknown"),
    ],
)
def test_parse_crs(element: etree.Element, expected: str):
    result = resolvers.parse_crs(element)
    assert result == expected


@pytest.mark.parametrize(
    "coordinates, expected",
    [
        ("1.0 2.0", (1.0, 2.0)),
        ("2.1,2.0", (2.1, 2.0)),
        ("3.1;1.0", (3.1, 1.0)),
        ("1.0   2.3", (1.0, 2.3)),  # Separated by multiple whitespaces
    ],
)
def test_parse_coordinates(coordinates: str, expected: tuple[float, float]):
    result = resolvers.parse_coordinates(coordinates)
    assert result == expected


@pytest.mark.parametrize(
    "invalid_coordinates",
    ["5.1|1.0", "1.0,,2.3", "1.0;;2.3"],
)
def test_parse_coordinates_invalid(invalid_coordinates: str):
    with pytest.raises(ValueError, match="Cannot parse coordinates"):
        resolvers.parse_coordinates(invalid_coordinates)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1.0", 1.0),
        (1, 1.0),
        (1.0, 1.0),
        (None, None),
        ([1, 2], None),
        ("[1, 2]", None),
    ],
)
def test_safe_float(value: Any, expected: float | None):
    result = resolvers.safe_float(value)
    assert result == expected


@pytest.mark.unittest
def test_process_bhrgt_data(bhrgt_no_namespace):
    """
    Tests resolvers.process_bhrgt_data for an XML element without namespaces. Behaviour
    for "real" BRO-like elements with data in namespaces is tested in test_xml_read
    module.

    """
    result = resolvers.process_bhrgt_data(
        bhrgt_no_namespace.getroot(),
        [
            "upperBoundary",
            "lowerBoundary",
            "geotechnicalSoilName",
            "nonExistingAttribute",  # Make sure non-existing attributes does not raise error
        ],
    )
    assert result == {
        "upperBoundary": ["0.0", "1.2", "2.1"],
        "lowerBoundary": ["1.2", "2.1", "2.5"],
        "geotechnicalSoilName": ["zand", "klei", "silt"],
        "nonExistingAttribute": [None, None, None],
    }

    # Test that at least "upperBoundary", "lowerBoundary" and "geotechnicalSoilName" are
    # retrieved.
    result = resolvers.process_bhrgt_data(bhrgt_no_namespace.getroot(), None)
    assert result == {
        "upperBoundary": ["0.0", "1.2", "2.1"],
        "lowerBoundary": ["1.2", "2.1", "2.5"],
        "geotechnicalSoilName": ["zand", "klei", "silt"],
    }


@pytest.mark.parametrize("element", [crs_element(), None])
def test_safe_get(element):
    result = resolvers.safe_get(element)
    assert result is None


@pytest.mark.unittest
def test_process_bhrp_data(testdatadir: Path):
    xml = etree.parse(testdatadir / r"xml/bhrp_bro.xml").getroot()
    layer_element = xml.find(
        "dispatchDocument/BHR_O/boreholeSampleDescription/bhrcommon:result",
        xml.nsmap,
    )
    result = resolvers.process_bhrp_data(layer_element, None)
    assert result == {
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

    result = resolvers.process_bhrp_data(
        layer_element,
        [
            "upperBoundary",
            "lowerBoundary",
            "standardSoilName",
            "containsGravel",
            "clayContent",
            "nonexistingAttribute",  # Make sure non-existing attributes does not raise error
        ],
    )
    assert result == {
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
        "containsGravel": ["nee", "nee", "nee", "nee", "nee", "nee"],
        "clayContent": ["30.0", "30.0", "32.0", "20.0", "4.0", "4.0"],
        "nonexistingAttribute": [None, None, None, None, None, None],
    }


@pytest.mark.unittest
def test_process_cpt_data(testdatadir: Path):
    xml = etree.parse(testdatadir / r"xml/cpt_bro.xml").getroot()
    layer_element = xml.find("dispatchDocument/CPT_O/conePenetrometerSurvey", xml.nsmap)
    result = resolvers.process_cpt_data(layer_element)
    assert isinstance(result, dict)
    assert_array_equal(
        list(result.keys()),
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


@pytest.mark.unittest
def test_strip_tag():
    el = etree.Element("{http://www.opengis.net/gml}Point")
    assert resolvers.strip_tag(el) == "point"

    el = etree.Element("Point")
    assert resolvers.strip_tag(el) == "point"

    el = etree.Element(
        "{http://www.opengis.net/gml}Point", nsmap={"gml": "http://www.opengis.net/gml"}
    )
    assert resolvers.strip_tag(el) == "point"


@pytest.mark.parametrize(
    "text, column_sep, row_sep",
    [
        ("1,2;3,4", ",", ";"),
        ("1 2,3 4", " ", ","),
        ("1;2:3:4", ";", ":"),
        ("1.2,3.4", ".", ","),
        ("1\t2\n3\t4", "\t", "\n"),
    ],
)
def test_textvalues_to_array(text, column_sep, row_sep):
    result = resolvers.textvalues_to_array(text, column_sep, row_sep)
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.unittest
def test_process_bhrg_data(testdatadir: Path):
    xml = etree.parse(testdatadir / r"xml/bhrg_bro.xml").getroot()
    layer_element = xml.find(
        "dispatchDocument/BHR_G_O/boreholeSampleDescription/bhrgcom:BoreholeSampleDescription/bhrgcom:descriptiveBoreholeLog/bhrgcom:DescriptiveBoreholeLog",
        xml.nsmap,
    )
    result = resolvers.process_bhrg_data(layer_element, None)
    assert result == {
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

    result = resolvers.process_bhrg_data(
        layer_element,
        [
            "upperBoundary",
            "lowerBoundary",
            "soilNameNEN5104",
            "carbonateContentClass",
            "constituentType",
            "nonexistingAttribute",  # Make sure non-existing attributes does not raise error
        ],
    )
    assert result == {
        "upperBoundary": ["0.000", "0.250", "1.600", "2.000", "2.500"],
        "lowerBoundary": ["0.250", "1.600", "2.000", "2.500", "3.000"],
        "soilNameNEN5104": [
            "zwakZandigeKlei",
            "sterkSiltigeKlei",
            "zwakZandigeKlei",
            "sterkZandigeKlei",
            "zwakSiltigZand",
        ],
        "carbonateContentClass": [
            "onbekend",
            "onbekend",
            "onbekend",
            "onbekend",
            "onbekend",
        ],
        "constituentType": ["puin", None, None, None, None],
        "nonexistingAttribute": [None, None, None, None, None],
    }


@pytest.mark.unittest
def test_process_bhrg_data(testdatadir: Path):
    xml = etree.parse(testdatadir / r"xml/sfr_bro.xml").getroot()
    layer_element = xml.find(
        "dispatchDocument/SFR_O/soilFaceDescription/sfrcom:SoilFaceDescription/sfrcom:soilProfile/sfrcom:SoilProfile",
        xml.nsmap,
    )
    result = resolvers.process_sfr_data(layer_element, None)
    assert result == {
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

    result = resolvers.process_sfr_data(
        layer_element,
        [
            "upperBoundary",
            "lowerBoundary",
            "soilNameNEN5104",
            "colour",
            "estimatedClayContent",
            "nonexistingAttribute",  # Make sure non-existing attributes does not raise error
        ],
    )
    assert result == {
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
        "colour": [
            "zwartBruin",
            "donkergrijs",
            "donkergrijs",
            "grijs",
            "grijs",
            None,
            None,
        ],
        "estimatedClayContent": ["28", "32", "48", "25", "20", "12", "8"],
        "nonexistingAttribute": [None, None, None, None, None, None, None],
    }
