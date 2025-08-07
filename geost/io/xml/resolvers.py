import re
from collections import defaultdict
from contextlib import suppress
from typing import Any

import numpy as np
from lxml import etree


def parse_crs(el: etree.Element) -> str:
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


def parse_coordinates(coords: str) -> tuple[float, float]:
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


def safe_float(value: Any) -> float | None:
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


def _process_layers(layers: list[etree.Element], attributes: list[str]) -> dict:
    """
    Process a list of XML elements representing layers and extract specified attributes.

    Parameters
    ----------
    layers : list[etree.Element]
        List of XML elements representing layers.
    attributes : list[str]
        List of attribute names to extract from each layer.

    Returns
    -------
    dict
        Dictionary with the searched layer attributes as keys and lists of each value per
        layer.

    """
    data = defaultdict(list)
    for layer in layers:
        for attr in attributes:
            attribute = layer.xpath(f".//*[local-name() = '{attr}']")
            value = safe_get(attribute[0]) if attribute else None
            data[attr].append(value)

    return data


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

    return _process_layers(layers, attributes)


def process_bhrg_data(el: etree.Element, attributes: list | None) -> dict:
    """
    Process an XML element containing the layer descriptions in BHR-G data objects.

    Parameters
    ----------
    el : etree.Element
        Element containing the layer descriptions.
    attributes : list[str] | None
        List with string names of the attributes to retrieve from each layer. If the input
        is None, it will be attempted to at least retrieve "upperBoundary", "lowerBoundary"
        and "soilNameNEN5104" from each layer.

    Returns
    -------
    dict
        Dictionary with the searched layer attributes as keys and lists of each value per
        layer.

    """
    if attributes is None:
        attributes = ["upperBoundary", "lowerBoundary", "soilNameNEN5104"]

    layers = el.xpath(".//*[local-name() = 'layer']")

    return _process_layers(layers, attributes)


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
        and "standardSoilName" from each layer.

    Returns
    -------
    dict
        Dictionary with the searched layer attributes as keys and lists of each value per
        layer.

    """
    if attributes is None:
        attributes = ["upperBoundary", "lowerBoundary", "standardSoilName"]

    layers = el.xpath(".//*[local-name() = 'soilLayer']")

    return _process_layers(layers, attributes)


def process_cpt_data(el: etree.Element) -> dict:
    """
    Process an XML element containing the data in CPT data objects.

    Parameters
    ----------
    el : etree.Element
        Element containing the CPT data.

    Returns
    -------
    dict
        Dictionary with the processed CPT data.

    """
    cpt_data = el.xpath(".//*[local-name() = 'cptResult']")[0]
    parameters = el.xpath(".//*[local-name() = 'parameters']")[0]

    encoding = cpt_data.xpath(".//*[local-name() = 'TextEncoding']")[0]
    row_sep = encoding.attrib.get("blockSeparator", ",")
    column_sep = encoding.attrib.get("tokenSeparator", ";")

    values = cpt_data.xpath(".//*[local-name() = 'values']")[0]
    values = textvalues_to_array(values.text, column_sep, row_sep)

    missing_value = -999999
    values[values == missing_value] = np.nan

    data = {}
    for ii, param in enumerate(parameters.iterchildren()):
        data[strip_tag(param)] = values[:, ii]

    return data


def process_sfr_data(el: etree.Element, attributes: list | None) -> dict:
    """
    Process an XML element containing the layer descriptions in SFR data objects.

    Parameters
    ----------
    el : etree.Element
        Element containing the layer descriptions.
    attributes : list[str] | None
        List with string names of the attributes to retrieve from each layer. If the input
        is None, it will be attempted to at least retrieve "upperBoundary", "lowerBoundary"
        and "soilNameNEN5104" from each layer.

    Returns
    -------
    dict
        Dictionary with the searched layer attributes as keys and lists of each value per
        layer.

    """
    if attributes is None:
        attributes = ["upperBoundary", "lowerBoundary", "soilNameNEN5104"]

    layers = el.xpath(".//*[local-name() = 'SoilLayer']")

    return _process_layers(layers, attributes)


def strip_tag(el: etree.Element) -> str:
    """
    Strip namespaces from the tag name of an XML element, returning only the local name.

    Parameters
    ----------
    el : etree.Element
        The XML element to strip.

    Returns
    -------
    str
        The local name of the element without any namespace prefix.

    """
    tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
    return tag.lower()


def textvalues_to_array(text: str, column_sep: str, row_sep: str) -> list:
    """
    Convert a string of text values into a list of lists, splitting by column and row
    separators.

    Parameters
    ----------
    text : str
        The input string containing text values.
    column_sep : str, optional
        The separator for columns (default is ',').
    row_sep : str, optional
        The separator for rows (default is ';').

    Returns
    -------
    list
        A list of lists containing the split text values.

    """
    ncols = text.split(row_sep)[0].count(column_sep) + 1

    if column_sep == ".":  # np.fromstring does not support '.' as a separator
        text = text.replace(column_sep, row_sep)
        sep = row_sep
    else:
        text = text.replace(row_sep, column_sep)
        sep = column_sep

    array = np.fromstring(text, sep=sep, dtype=float)

    return array.reshape(-1, ncols)
