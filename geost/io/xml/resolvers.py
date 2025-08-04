import re
from collections import defaultdict
from contextlib import suppress
from typing import Any

from lxml import etree


def parse_crs(el: etree.Element, **_) -> str:
    """
    Retrieve the coordinate reference system (CRS) from an XML element.

    Parameters
    ----------
    el : etree.Element
        The XML element containing the CRS information.

    Returns
    -------
    str
        String representing the CRS (e.g. 'urn:ogc:def:crs:EPSG::28992'), or "unknown"
        if not found.

    """
    return el.attrib.get("srsName", "unknown")


def parse_coordinates(coords: str, **_) -> tuple[float, float]:
    """
    Parse a string containing coordinates as a tuple of floats.

    Parameters
    ----------
    coords : str
        Any of {'x y', 'x,y', 'x;y'} where x and y are parsable by float

    Returns
    -------
    tuple[float, float]
        The parsed coordinates as a tuple of floats.

    """
    if " " in coords:
        splitter = r"\s+"
    elif "," in coords:
        splitter = ","
    elif ";" in coords:
        splitter = ";"
    else:
        raise ValueError(f"Cannot parse coordinates: '{coords}'")

    try:
        x, y = re.split(splitter, coords)
    except ValueError as e:
        raise ValueError(f"Cannot parse coordinates: '{coords}'") from e

    return float(x), float(y)


def safe_float(value: Any, **_) -> float | None:
    """
    Try to cast a value to a float dtype. Returns None if the value cannot be casted.

    Parameters
    ----------
    value : Any
        The value to cast.

    Returns
    -------
    float | None
        Value as a float type or None if the value cannot be casted into a float.

    """
    with suppress(TypeError, ValueError):
        return float(value)


def clean_string(val: str) -> str:
    """
    Clean a string to keep only letters, numbers and punctuation.

    Parameters
    ----------
    val : str
        The input string to clean.

    Returns
    -------
    str
        The cleaned string with unwanted characters removed.

    """
    return re.sub(r"[^\w.,:;!?()-]", "", val)


def safe_get(el: etree.Element):
    """
    Safely get the text from an `etree.Element` instance. Returns `None` if it would raise
    an `AttributeError`. This occurs for example when `el` is a NoneType.

    """
    with suppress(AttributeError):
        value = el.text
        if value is None:
            return None
        else:
            return clean_string(value)


def process_bhrgt_data(el: etree.Element, attributes: list | None) -> dict:
    """
    Process an XML element containing the layer descriptions in BHR-GT data objects.

    Parameters
    ----------
    el : etree.Element
        Element containing the layer descriptions.
    attributes : list[str] | None
        List with string names of the attributes to retrieve from each layer. If the input
        is None, it will be attempted to at least retrieve "upperBoundary", "lowerBoundary"
        and "geotechnicalSoilName" from each layer.

    Returns
    -------
    dict
        Dictionary with the searched layer attributes as keys and lists of each value per
        layer.

    """
    if attributes is None:
        attributes = ["upperBoundary", "lowerBoundary", "geotechnicalSoilName"]

    layers = el.xpath(".//*[local-name() = 'layer']")

    data = defaultdict(list)
    for layer in layers:
        for attr in attributes:
            attribute = layer.xpath(f".//*[local-name() = '{attr}']")
            value = safe_get(attribute[0]) if attribute else None
            data[attr].append(value)

    return data


def process_bhrp_data(el: etree.Element, attributes: list | None) -> dict:
    """
    Process an XML element containing the layer descriptions in BHR-P data objects.

    Parameters
    ----------
    el : etree.Element
        Element containing the layer descriptions.
    attributes : list[str] | None
        List with string names of the attributes to retrieve from each layer. If the input
        is None, it will be attempted to at least retrieve "upperBoundary", "lowerBoundary"
        and "geotechnicalSoilName" from each layer.

    Returns
    -------
    dict
        Dictionary with the searched layer attributes as keys and lists of each value per
        layer.

    """
    if attributes is None:
        attributes = ["upperBoundary", "lowerBoundary", "standardSoilName"]

    layers = el.xpath(".//*[local-name() = 'soilLayer']")

    data = defaultdict(list)
    for layer in layers:
        for attr in attributes:
            attribute = layer.xpath(f".//*[local-name() = '{attr}']")
            value = safe_get(attribute[0]) if attribute else None
            data[attr].append(value)

    return data
