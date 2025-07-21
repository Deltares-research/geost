import pytest
from lxml import etree

from geost.io.xml import resolvers


def crs_element():
    el = etree.Element("Point", srsName="urn:ogc:def:crs:EPSG::28992")
    return el


def empty_crs_element():
    el = etree.Element("Point")
    return el


@pytest.mark.parametrize(
    "element, expected",
    [
        (crs_element(), "urn:ogc:def:crs:EPSG::28992"),
        (empty_crs_element(), "unknown"),
    ],
)
def test_parse_crs(element, expected):
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
def test_parse_coordinates(coordinates, expected):
    result = resolvers.parse_coordinates(coordinates)
    assert result == expected


@pytest.mark.parametrize(
    "invalid_coordinates",
    ["5.1|1.0", "1.0,,2.3", "1.0;;2.3"],
)
def test_parse_coordinates_invalid(invalid_coordinates):
    with pytest.raises(ValueError, match="Cannot parse coordinates"):
        resolvers.parse_coordinates(invalid_coordinates)
