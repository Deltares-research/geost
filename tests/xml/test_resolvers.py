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


def create_layer_element(root: etree.Element, top: float, bot: float, lith: str):
    el = etree.SubElement(root, "layer")

    top_ = etree.SubElement(el, "upperBoundary")
    top_.text = top

    bot_ = etree.SubElement(el, "lowerBoundary")
    bot_.text = bot

    soil = etree.SubElement(el, "soil")
    name = etree.SubElement(soil, "geotechnicalSoilName")
    name.text = lith
    return el


@pytest.fixture
def bhrgt_no_namespaces():
    root = etree.Element("descriptiveBoreholeLog")

    tops = [0.0, 1.2, 2.1]
    bottoms = [1.2, 2.1, 2.5]
    lithologies = ["zand", "klei", "silt"]

    for top, bot, lith in zip(tops, bottoms, lithologies):
        create_layer_element(root, str(top), str(bot), lith)
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


@pytest.mark.xfail(reason="Double ',,' or ';;' does not lead to expected error.")
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


@pytest.mark.skip(reason="Test appears to break testing suite, check fixture value.")
def test_process_bhrgt_data(bhrgt_no_namespaces):
    """
    Tests `resolvers.process_bhrgt_data` for an XML element without namespaces. Behaviour
    for "real" BRO-like elements with data in namespaces is tested in `test_xml_read`
    module.

    """
    result = resolvers.process_bhrgt_data(
        bhrgt_no_namespaces,
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
    result = resolvers.process_bhrgt_data(bhrgt_no_namespaces, None)
    assert result == {
        "upperBoundary": ["0.0", "1.2", "2.1"],
        "lowerBoundary": ["1.2", "2.1", "2.5"],
        "geotechnicalSoilName": ["zand", "klei", "silt"],
    }


@pytest.mark.parametrize("element", [crs_element(), None])
def test_safe_get(element):
    result = resolvers.safe_get(element)
    assert result is None
