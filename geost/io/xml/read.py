"""
The basic XML parsing design (i.e. `_read_xml`) in this module is based on Pygef's approach
but the aim of this module is to provide a more flexible and extensible approach to parsing
XML files.

See: https://github.com/cemsbv/pygef/tree/master/src/pygef/broxml for the original design.

"""

import io
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from lxml import etree

from . import schemas

_BaseParser = etree.XMLParser(
    resolve_entities=False, dtd_validation=False
)  # Speeds up parsing by not resolving entities or validating DTDs.


def read(
    files: str | Path | io.StringIO | Iterable[str | Path],
    reader: Callable,
    **kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read one or multiple XML files or string objects and create `geost` compatible separate
    `header` and `data` DataFrames.

    Parameters
    ----------
    files : str | Path | io.StringIO | Iterable[str | Path]
        List of XML files to process.
    reader : Callable
        Function to read the XML data. See `geost.io.xml.READERS` for available XML readers.
    **kwargs : dict[str, Any]
        Additional keyword arguments to pass to the reader function.

    Returns
    -------
    tuple of pandas.DataFrame
        Tuple containing the header DataFrame and the data DataFrame.

    """
    if isinstance(files, (str, Path, io.StringIO)):
        files = [files]

    header = []
    data = []
    lengths = []

    for file in files:
        xml_data = reader(file, **kwargs)
        xml_data["x"], xml_data["y"] = xml_data.pop("location")
        df = pd.DataFrame(xml_data.pop("data"))

        header.append(xml_data)
        data.append(df)
        lengths.append(len(df))

    header = pd.DataFrame(header)
    header["end"] = header["surface"] - header["end"]
    data = pd.concat(data, ignore_index=True)

    # Add relevant header attributes to the data DataFrame
    attributes_for_data = ["nr", "x", "y", "surface", "end"]
    header_data = header.loc[header.index.repeat(lengths), attributes_for_data]
    data = pd.concat([header_data.reset_index(drop=True), data], axis=1)

    if "upperBoundary" in data.columns or "lowerBoundary" in data.columns:
        data.rename(
            columns={"upperBoundary": "top", "lowerBoundary": "bottom"}, inplace=True
        )
        data[["top", "bottom"]] = data[["top", "bottom"]].astype(float)

    return header, data


def _read_xml(
    file: bytes | str | Path,
    schema: dict[str, Any],
    payload_root: str = None,
    read_all: bool = False,
) -> dict | list[dict]:
    if isinstance(file, bytes):
        root = etree.fromstring(file, parser=_BaseParser)
    else:
        root = etree.parse(file, parser=_BaseParser).getroot()

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

                    if attributes := d.get("layer-attributes", ""):
                        data[key] = func(el, attributes=attributes)
                    else:
                        data[key] = func(el)
                else:
                    data[key] = el.text
            else:
                data[key] = None

        if read_all:
            output.append(data)
        else:
            if len(payloads) > 1:
                warnings.warn(
                    "Multiple payloads found in XML but 'read_all' is set to False. "
                    "Set 'read_all' to True to read all payloads."
                )
            return data

    return output


def read_bhrgt(
    file: bytes | str | Path,
    company: str = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> dict | list[dict]:
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
    read_all : bool, optional
        Generally, each XML file contains data for a single borehole but in some cases
        multiple boreholes can be present in the XML file. If set to True, all boreholes
        will be read from the XML file. The default is False, which reads only the first.
        A warning will be raised if multiple boreholes are found in the XML file and
        read_all is False.

    Returns
    -------
    dict | list[dict]
        Dict or list of dictionaries containing the extracted data from the XML file. If
        the XML file contains data for a single borehole, a dictionary will be returned.
        If multiple boreholes are found and `read_all` is True, a list of dictionaries
        will be returned.

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
            schema = schemas.bhrgt[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}' in BHR-GT. Supported companies are: "
                f"{schemas.bhrgt.keys()}. Define a custom schema or use a supported company."
            ) from e

    try:
        result = _read_xml(file, schema, schema.get("payload_root", None), read_all)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid xml schema for XML file: {file}.") from e

    return result


def read_bhrp(
    file: bytes | str | Path,
    company: str = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> dict | list[dict]:
    """
    Read borehole data (BHR-P) from an XML file or bytestring and extract relevant data
    based on a predefined or custom schema that describes the XML structure.

    Parameters
    ----------
    file : bytes | str | Path
        XML filepath-like or bytestring containing the XML data to read.
    company : str, optional
        Specify a company name to use a predefined schema for that company if available.
        See `bhrp.SCHEMA` for available companies. The default is None, then company will
        default to "BRO" and the predefined BRO schema will be used if no custom schema
        is defined.
    schema : dict[str, Any], optional
        Custom schema used to parse the XML structure.
    read_all : bool, optional
        Generally, each XML file contains data for a single borehole but in some cases
        multiple boreholes can be present in the XML file. If set to True, all boreholes
        will be read from the XML file. The default is False, which reads only the first.
        A warning will be raised if multiple boreholes are found in the XML file and
        read_all is False.

    Returns
    -------
    dict | list[dict]
        Dict or list of dictionaries containing the extracted data from the XML file. If
        the XML file contains data for a single borehole, a dictionary will be returned.
        If multiple boreholes are found and `read_all` is True, a list of dictionaries
        will be returned.

    Raises
    ------
    ValueError
        If no predefined schema is found for the specified company.
    SyntaxError
        If the XML file does not conform to the expected schema.

    """
    company = company or "BRO"

    if schema is None:
        try:
            schema = schemas.bhrp[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}' in BHR-P. Supported companies are: "
                f"{schemas.bhrp.keys()}. Define a custom schema or use a supported company."
            ) from e

    try:
        result = _read_xml(file, schema, schema.get("payload_root", None), read_all)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid xml schema for XML file: {file}.") from e

    return result


def read_cpt(
    file: bytes | str | Path,
    company: str = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> dict | list[dict]:
    """
    Read Cone Penetration Test (CPT) data from an XML file or bytestring and extract
    relevant data based on a predefined or custom schema that describes the XML structure.

    Parameters
    ----------
    file : bytes | str | Path
        XML filepath-like or bytestring containing the XML data to read.
    company : str, optional
        Specify a company name to use a predefined schema for that company if available.
        See `cpt.SCHEMA` for available companies. The default is None, then company will
        default to "BRO" and the predefined BRO schema will be used if no custom schema
        is defined.
    schema : dict[str, Any], optional
        Custom schema used to parse the XML structure.
    read_all : bool, optional
        Generally, each XML file contains data for a single CPT but in some cases
        multiple CPTs can be present in the XML file. If set to True, all CPTs will be
        read from the XML file. The default is False, which reads only the first.
        A warning will be raised if multiple CPTs are found in the XML file and
        read_all is False.

    Returns
    -------
    dict | list[dict]
        Dict or list of dictionaries containing the extracted data from the XML file. If
        the XML file contains data for a single CPT, a dictionary will be returned.
        If multiple CPTs are found and `read_all` is True, a list of dictionaries
        will be returned.

    Raises
    ------
    ValueError
        If no predefined schema is found for the specified company.
    SyntaxError
        If the XML file does not conform to the expected schema.

    """
    company = company or "BRO"

    if schema is None:
        try:
            schema = schemas.cpt[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}' in CPT. Supported companies are: "
                f"{schemas.cpt.keys()}. Define a custom schema or use a supported company."
            ) from e

    try:
        result = _read_xml(file, schema, schema.get("payload_root", None), read_all)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid xml schema for XML file: {file}.") from e

    return result


def read_bhrg(
    file: bytes | str | Path,
    company: str = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> dict | list[dict]:
    """
    Read Geological borehole data (BHR-G) from an XML file or bytestring and extract
    relevant data based on a predefined or custom schema that describes the XML structure.

    Parameters
    ----------
    file : bytes | str | Path
        XML filepath-like or bytestring containing the XML data to read.
    company : str, optional
        Specify a company name to use a predefined schema for that company if available.
        See `schemas.bhrg` for available companies. The default is None, then company will
        default to "BRO" and the predefined BRO schema will be used if no custom schema
        is defined.
    schema : dict[str, Any], optional
        Custom schema used to parse the XML structure.
    read_all : bool, optional
        Generally, each XML file contains data for a single borehole but in some cases
        multiple boreholes can be present in the XML file. If set to True, all boreholes
        will be read from the XML file. The default is False, which reads only the first.
        A warning will be raised if multiple boreholes are found in the XML file and
        read_all is False.

    Returns
    -------
    dict | list[dict]
        Dict or list of dictionaries containing the extracted data from the XML file. If
        the XML file contains data for a single borehole, a dictionary will be returned.
        If multiple boreholes are found and `read_all` is True, a list of dictionaries
        will be returned.

    Raises
    ------
    ValueError
        If no predefined schema is found for the specified company.
    SyntaxError
        If the XML file does not conform to the expected schema.

    """
    company = company or "BRO"

    if schema is None:
        try:
            schema = schemas.bhrg[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}' in BHR-G. Supported companies are: "
                f"{schemas.bhrg.keys()}. Define a custom schema or use a supported company."
            ) from e

    try:
        result = _read_xml(file, schema, schema.get("payload_root", None), read_all)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid xml schema for XML file: {file}.") from e

    return result


def read_sfr(
    file: bytes | str | Path,
    company: str = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> dict | list[dict]:
    """
    Read Soil outcrop descriptions (SFR) from an XML file or bytestring and extract
    relevant data based on a predefined or custom schema that describes the XML structure.

    Parameters
    ----------
    file : bytes | str | Path
        XML filepath-like or bytestring containing the XML data to read.
    company : str, optional
        Specify a company name to use a predefined schema for that company if available.
        See `schemas.sfr` for available companies. The default is None, then company will
        default to "BRO" and the predefined BRO schema will be used if no custom schema
        is defined.
    schema : dict[str, Any], optional
        Custom schema used to parse the XML structure.
    read_all : bool, optional
        Generally, each XML file contains data for a single outcrop but in some cases
        multiple outcrops can be present in the XML file. If set to True, all outcrops
        will be read from the XML file. The default is False, which reads only the first.
        A warning will be raised if multiple outcrops are found in the XML file and
        read_all is False.

    Returns
    -------
    dict | list[dict]
        Dict or list of dictionaries containing the extracted data from the XML file. If
        the XML file contains data for a single outcrop, a dictionary will be returned.
        If multiple outcrops are found and `read_all` is True, a list of dictionaries
        will be returned.

    Raises
    ------
    ValueError
        If no predefined schema is found for the specified company.
    SyntaxError
        If the XML file does not conform to the expected schema.

    """
    company = company or "BRO"

    if schema is None:
        try:
            schema = schemas.sfr[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}' in SFR. Supported companies are: "
                f"{schemas.sfr.keys()}. Define a custom schema or use a supported company."
            ) from e

    try:
        result = _read_xml(file, schema, schema.get("payload_root", None), read_all)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid xml schema for XML file: {file}.") from e

    return result
