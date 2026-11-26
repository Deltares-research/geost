import sqlite3
import zipfile
from contextlib import closing
from xml.etree import ElementTree as ET

import pytest

from geost.download import pdok_geopackage as pdok
from geost.download.pdok_geopackage import get_geopackage_url


class FakeResponse:
    def __init__(self, content: bytes = b""):
        self.content = content

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_gpkg(path):
    with closing(sqlite3.connect(path)) as con:
        con.execute(f"PRAGMA application_id = {int(pdok.GPKGApplicationIDs.GPKG)}")
        con.execute("CREATE TABLE dummy (id INTEGER)")
        con.commit()
    return path


def make_fk_gpkg(path):
    with closing(sqlite3.connect(path)) as con:
        con.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        con.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, "
            "parent_id INTEGER REFERENCES parent(id))"
        )
        con.execute(
            "CREATE TABLE gpkg_skipped (id INTEGER PRIMARY KEY, "
            "parent_id INTEGER REFERENCES parent(id))"
        )
        con.commit()
    return path


def index_names(path):
    with closing(sqlite3.connect(path)) as con:
        return {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }


@pytest.fixture
def gpkg_bytes(tmp_path):
    return make_gpkg(tmp_path / "source.gpkg").read_bytes()


@pytest.fixture
def patch_download(monkeypatch):
    def _patch(url, content):
        monkeypatch.setattr(pdok, "get_geopackage_url", lambda key: url)
        monkeypatch.setattr(
            pdok.requests, "get", lambda *a, **kw: FakeResponse(content)
        )

    return _patch


@pytest.mark.unittest
def test_get_geopackage_url_bhrgt():
    result = get_geopackage_url("BHR-GT")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/geotechnischbooronderzoek.zip")


@pytest.mark.unittest
def test_get_geopackage_url_bhrg():
    result = get_geopackage_url("BHR-G")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brobhrgvolledigeset.gpkg")


@pytest.mark.unittest
def test_get_geopackage_url_bhrp():
    result = get_geopackage_url("BHR-P")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brobhrpvolledigeset.zip")


@pytest.mark.unittest
def test_get_geopackage_url_cpt():
    result = get_geopackage_url("CPT")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brocptvolledigeset_v2_0.zip")


@pytest.mark.unittest
def test_get_geopackage_url_sfr():
    result = get_geopackage_url("SFR")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brosfrvolledigeset.zip")


@pytest.mark.unittest
def test_get_geopackage_url_invalid_type():
    with pytest.raises(KeyError, match="Unknown dataset type: invalid_type"):
        get_geopackage_url("invalid_type")


@pytest.mark.unittest
def test_alternate_links():
    root = ET.fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        '<entry><link rel="alternate" href="a"/><link rel="self" href="b"/></entry>'
        '<entry><link rel="alternate" href="c"/></entry>'
        "</feed>"
    )
    assert [link.get("href") for link in pdok._alternate_links(root)] == ["a", "c"]


@pytest.mark.unittest
def test_get_xml(monkeypatch):
    monkeypatch.setattr(
        pdok.requests, "get", lambda *a, **kw: FakeResponse(b"<root><child/></root>")
    )
    root = pdok._get_xml("https://example.com/feed.xml")
    assert root.tag == "root"
    assert root.find("child") is not None


@pytest.mark.unittest
def test_download_url(monkeypatch, tmp_path):
    content = b"0123456789" * 10
    monkeypatch.setattr(pdok.requests, "get", lambda *a, **kw: FakeResponse(content))
    outfile = tmp_path / "file.bin"

    result = pdok._download_url("https://example.com/file.bin", outfile, 10, 7)

    assert result == outfile
    assert outfile.read_bytes() == content
    assert not outfile.with_name("file.bin.part").exists()


@pytest.mark.unittest
def test_create_bro_index(tmp_path):
    gpkg = make_fk_gpkg(tmp_path / "test.gpkg")
    pdok.create_bro_index(gpkg, "CREATE INDEX idx_test ON child (parent_id)")
    assert "idx_test" in index_names(gpkg)


@pytest.mark.unittest
def test_write_fk_index(tmp_path):
    gpkg = make_fk_gpkg(tmp_path / "test.gpkg")
    pdok.write_fk_index(gpkg)
    names = index_names(gpkg)
    assert "idx_child_id" in names
    assert not any("gpkg_skipped" in n for n in names)


@pytest.mark.unittest
def test_download_gpkg(patch_download, gpkg_bytes, tmp_path):
    patch_download("https://example.com/atom/downloads/data.gpkg", gpkg_bytes)
    savedir = tmp_path / "new" / "dir"

    result = pdok.download("CPT", savedir)

    assert result == savedir / "data.gpkg"
    assert result.exists()


@pytest.mark.unittest
@pytest.mark.parametrize("delete_zip", [True, False])
def test_download_zip(patch_download, gpkg_bytes, tmp_path, delete_zip):
    zip_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("data.gpkg", gpkg_bytes)
    patch_download("https://example.com/atom/downloads/data.zip", zip_path.read_bytes())
    savedir = tmp_path / "out"

    result = pdok.download("CPT", savedir, delete_zip=delete_zip)

    assert result == savedir / "data.gpkg"
    assert result.exists()
    assert (savedir / "data.zip").exists() is not delete_zip
