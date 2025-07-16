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
        splitter = " "
    elif "," in coords:
        splitter = ","
    elif ";" in coords:
        splitter = ";"
    else:
        raise ValueError(f"Cannot parse coordinates: '{coords}'")
    x, y = coords.split(splitter)
    return float(x), float(y)
