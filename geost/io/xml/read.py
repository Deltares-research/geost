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

                    data[key] = func(el, namespaces=payload.nsmap)
                else:
                    data[key] = el.text
            else:
                data[key] = None

        output.append(data)

    return output


def read_bhrgt(file: str | Path, company: str = None, schema: dict[str, Any] = None):
    company = company or "BRO"

    if schema is None:
        try:
            schema = bhrgt.SCHEMA[company]
        except KeyError as e:
            raise ValueError(
                f"No predefined schema for '{company}'. Supported companies are: "
                f"{bhrgt.SCHEMA.keys()}. Define a custom schema or use a supported company."
            ) from e

    if isinstance(file, str) and not Path(file).exists():
        root = etree.fromstring(file, parser=_BaseParser).getroot()
    else:
        root = etree.parse(file, parser=_BaseParser).getroot()

    return _read_xml(root, schema, schema.get("payload_root", None))
