from pathlib import Path
from typing import Any

import pytest
from lxml import etree

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
