from pathlib import Path

import pytest

from geost.io import xml


@pytest.fixture
def xml_dir():
    return Path(__file__).parents[1] / "data" / "xml"


@pytest.fixture
def xml_string(xml_dir):
    with open(xml_dir / "bhrgt_bro.xml", "rb") as file:
        string = file.read()
    return string


@pytest.fixture
def custom_bhrgt_schema():
    return {
        "payload_root": "dispatchDocument",
        "bro_id": {"xpath": "brocom:broId"},
        "date": {"xpath": "reportHistory/reportStartDate/brocom:date"},
    }


@pytest.mark.unittest
def test_read_bhrgt(xml_dir):
    bro_xml = xml_dir / "bhrgt_bro.xml"

    data = xml.read_bhrgt(bro_xml)

    assert isinstance(data, list)
    assert len(data) == 1

    core = data[0]
    assert isinstance(core, dict)
    assert core["nr"] == "BHR000000336600"

    with pytest.raises(SyntaxError, match="Invalid xml schema"):
        xml.read_bhrgt(bro_xml, company="Wiertsema")

    with pytest.raises(ValueError, match="No predefined schema for 'UnknownCompany'"):
        xml.read_bhrgt(bro_xml, company="UnknownCompany")


@pytest.mark.unittest
def test_read_bhrgt_with_custom_schema(xml_dir, custom_bhrgt_schema):
    data = xml.read_bhrgt(xml_dir / "bhrgt_bro.xml", schema=custom_bhrgt_schema)
    assert isinstance(data, list)
    assert data[0] == {"bro_id": "BHR000000336600", "date": "2020-04-15"}


@pytest.mark.unittest
def test_read_bhrgt_from_string(xml_string):
    data = xml.read_bhrgt(xml_string)
    assert isinstance(data, list)
    assert data[0]["nr"] == "BHR000000336600"


@pytest.mark.unittest
def test_read_bhrgt_with_custom_schema_no_payload_root(xml_dir):
    """
    Test reading an xml string without a payload root. This can lead to unexpected results
    as the XML root can contain multiple elements with additional information besides the
    main data element.

    """
    schema = {"bro_id": {"xpath": "BHR_GT_O/brocom:broId"}}
    data = xml.read_bhrgt(xml_dir / "bhrgt_bro.xml", schema=schema)
    assert isinstance(data, list)
    assert data[0] == {"bro_id": None}
    assert data[1] == {"bro_id": None}
    assert data[2] == {"bro_id": None}
    assert data[3] == {"bro_id": "BHR000000336600"}
