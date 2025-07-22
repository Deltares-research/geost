"""
The XML parsing design in this module is based on Pygef's approach but the aim of this module
is to provide a more flexible and extensible approach to parsing XML files.

See: https://github.com/cemsbv/pygef/tree/master/src/pygef/broxml for the original design.

"""

from pathlib import Path
from typing import Any

from lxml import etree

from .schemas import bhrgt

_BaseParser = etree.XMLParser(
    resolve_entities=False, dtd_validation=False
)  # Speeds up parsing by not resolving entities or validating DTDs.


def _read_xml(
    root: etree.Element, schema: dict[str, Any], payload_root: str = None
) -> list[dict]:
    if payload_root is not None:
        xmldata = root.find(payload_root, root.nsmap)

        if xmldata is None:
            raise ValueError(f"Payload root '{payload_root}' not found in XML.")
    else:
        xmldata = root  # Try to parse the entire root using the given schema.

    output = []

    payloads = xmldata.findall("./*")
    for payload in payloads:
        data = {}

        for key, d in schema.items():
            if key == "payload_root":
                continue
            el = payload.find(d["xpath"], payload.nsmap)

            if el is not None:
                if "resolver" in d:
                    func = d["resolver"]

                    if "el-attr" in d:
                        el = getattr(el, d["el-attr"])

                    data[key] = func(el, attributes=d.get("layer-attributes", ""))
                else:
                    data[key] = el.text
            else:
                data[key] = None

        output.append(data)

    return output


def read_bhrgt(
    file: bytes | str | Path, company: str = None, schema: dict[str, Any] = None
) -> list[dict]:
    """
    Read Geotechnical borehole data (BHR-GT) from an XML file or bytestring and extract
    relevant data based on a predefined or custom schema that describes the XML structure.

    Parameters
    ----------
    file : bytes | str | Path
        XML filepath-like or bytestring containing the XML data to read.
    company : str, optional
        Specify a company name to use a predefined schema for that company if available.
        See `bhrgt.SCHEMA` for available companies. The default is None, then company will
        default to "BRO" and the predefined BRO schema will be used if no custom schema
        is defined.
    schema : dict[str, Any], optional
        Custom schema used to parse the XML structure.

    Returns
    -------
    list[dict]
        List of dictionaries containing the extracted data from the XML file. If the XML
        file contains data for a single

    Raises
    ------
    ValueError
        Will raise a ValueError if no custom `schema` is defined and no predefined schema
        is available for the company.
    SyntaxError
        Raises a SyntaxError if the XML file does not conform to the expected schema. For
        example, with incorrect namespaces in the schema.

    Examples
    --------
    Read a BHR-GT XML for a specific company:
    >>> data = read_bhrgt("bhrgt_file.xml", company="Wiertsema") # Read for specific company

    Read a BHR-GT XML using a custom schema:
    >>> custom_schema = {
    ...     "payload_root": "dispatchDocument",
    ...     "bro_id": {"xpath": "brocom:broId"},
    ...     "date": {"xpath": "reportHistory/reportStartDate/brocom:date"}
    ...     "coordinates": {
    ...         "xpath": "./deliveredLocation/bhrgtcom:location/gml:Point/gml:pos",
    ...         "resolver": resolvers.parse_coordinates, # Add a custom function to parse coordinates
    ...         "el-attr": "text"  # Use the text attribute of the element
    ...     }
    ... }
    >>> data = read_bhrgt("bhrgt_file.xml", schema=custom_schema)  # Use custom schema

    Read a BHR-GT XML directly via a BRO API response:
    >>> from geost.bro import BroApi
    >>> api = BroApi()
    >>> response = api.get_objects("BHR000000449853", "BHR-GT") # Returns a list of XML strings
    >>> data = read_bhrgt(response[0])

    """
    company = company or "BRO"

    if schema is None:
        try:
            schema = bhrgt.SCHEMA[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}'. Supported companies are: "
                f"{bhrgt.SCHEMA.keys()}. Define a custom schema or use a supported company."
            ) from e

    if isinstance(file, bytes):
        root = etree.fromstring(file, parser=_BaseParser)
    else:
        root = etree.parse(file, parser=_BaseParser).getroot()

    try:
        result = _read_xml(root, schema, schema.get("payload_root", None))
    except SyntaxError as e:
        raise SyntaxError(
            f"Invalid xml schema for XML file: {file}. Cannot use predefined schema "
            f"for company '{company}'. Specify correct company or use a custom schema."
        ) from e

    return result
