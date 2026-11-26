import sqlite3
import warnings
import zipfile
from contextlib import closing
from enum import IntEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests

ATOM_URLS = {
    "BHR-GT": "https://service.pdok.nl/tno/bro-geotechnisch-booronderzoek/atom/index.xml",
    "BHR-G": "https://service.pdok.nl/tno/bro-geologisch-booronderzoek/atom/index.xml",
    "BHR-P": "https://service.pdok.nl/tno/bro-bodemkundig-booronderzoek/atom/index.xml",
    "CPT": "https://service.pdok.nl/tno/bro-geotechnischsondeeronderzoek/atom/index.xml",
    "SFR": "https://service.pdok.nl/tno/bro-wandonderzoek/atom/index.xml",
}

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ATOM_MIME = "application/atom+xml"


class GPKGApplicationIDs(IntEnum):
    GPKG = 0x47504B47
    GP10 = 0x47503130
    GP11 = 0x47503131


def download(
    type_: Literal["BHR-GT", "BHR-G", "BHR-P", "CPT", "SFR"],
    savedir: str | Path,
    delete_zip: bool = True,
    chunk_size_mb: int = 1 << 20,
    timeout: int = 60,
):
    """
    Download a BRO GeoPackage from PDOK for the given dataset type.

    The download link is looked up in the PDOK ATOM feed. If the downloaded file is a
    zip archive, the .gpkg file is extracted from it.

    Parameters
    ----------
    type_ : {"BHR-GT", "BHR-G", "BHR-P", "CPT", "SFR"}
        Dataset type to download: geotechnical boreholes (BHR-GT), geological boreholes
        (BHR-G), pedological boreholes (BHR-P), cone penetration tests (CPT) or
        soil profile wall investigations (SFR).
    savedir : str | Path
        Directory to save the file to. It is created if it does not exist.
    delete_zip : bool, optional
        Whether to delete the zip file after extracting the GeoPackage. Only relevant if
        the download is a zip file. The default is True.
    chunk_size_mb : int, optional
        Size of the chunks used while streaming the download. Note that despite the name,
        the value is passed as a number of bytes. The default is 1 << 20 (1 MiB).
    timeout : int, optional
        Timeout in seconds for the download request. The default is 60.

    Returns
    -------
    Path
        Path to the downloaded GeoPackage.

    Raises
    ------
    KeyError
        If ``type_`` is not a known dataset type.
    ValueError
        If the downloaded zip file does not contain a .gpkg file.

    Warns
    -----
    UserWarning
        If the file is not recognized as a GeoPackage 1.0, 1.1 or 1.2+ file.

    Examples
    --------
    Download the CPT GeoPackage to a local folder:

    >>> gpkg = download("CPT", "data/bro")

    """
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    url = get_geopackage_url(type_)

    download_file = Path(urlparse(url).path)
    downloaded = _download_url(
        url, savedir / download_file.name, timeout=timeout, chunk_size_mb=chunk_size_mb
    )

    if zipfile.is_zipfile(downloaded):
        with zipfile.ZipFile(downloaded, "r") as z:
            for info in z.infolist():
                if info.filename.endswith(".gpkg"):
                    print("Extracting zip file")
                    z.extract(info, savedir)
                    gpkg = savedir / info.filename
                    break
            else:
                raise ValueError("No .gpkg found in zip")

        if delete_zip:
            downloaded.unlink()
    else:
        gpkg = downloaded

    if not is_valid(gpkg):
        warnings.warn(
            "Geopackage is not recognized as one of GeoPackage 1.0, 1.1 or 1.2+ formats."
        )

    return gpkg


def get_geopackage_url(key: str) -> str:
    """
    Retrieve the download link (.gpkg or .zip) for a key in ``ATOM_URLS``.

    """
    if key not in ATOM_URLS:
        raise KeyError(f"Unknown dataset type: {key}. Must be one of {list(ATOM_URLS)}")

    top = _get_xml(ATOM_URLS[key])

    # Step 1: go from top feed to dataset feed
    dataset_url = next(
        link.get("href")
        for link in _alternate_links(top)
        if link.get("type") == ATOM_MIME
    )

    dataset = _get_xml(dataset_url)
    return next(
        link.get("href")
        for link in _alternate_links(dataset)
        if link.get("type") != ATOM_MIME
    )


def _alternate_links(root):
    """
    Find all rel="alternate"-links in the feed entries.

    """
    for entry in root.findall(f"{ATOM_NS}entry"):
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("rel") == "alternate":
                yield link


def _get_xml(url, timeout=30) -> ET.Element:
    """
    Fetch the XML document from the ATOM feed URL.

    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def is_valid(path: str | Path) -> bool:
    """
    Check whether a file is a GeoPackage by reading its SQLite application ID.

    """
    with closing(sqlite3.connect(path)) as con:
        application_id = con.execute("PRAGMA application_id").fetchone()[0]
    return application_id in GPKGApplicationIDs


def _download_url(url: str, outfile: Path, timeout: int, chunk_size_mb: int) -> Path:
    """
    Stream a URL to ``outfile`` via a temporary .part file and return ``outfile``.

    """
    tmp = outfile.with_name(f"{outfile.name}.part")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size_mb):
                f.write(chunk)

    tmp.replace(outfile)
    return outfile


def create_bro_index(gpkg: str | Path, sql: str) -> None:
    """
    Execute a (CREATE INDEX) SQL statement on the GeoPackage and update statistics.

    """
    with closing(sqlite3.connect(gpkg)) as con:
        con.execute(sql)
        con.execute("ANALYZE")
        con.commit()


def write_fk_index(gpkg) -> None:
    """
    Create an index on every foreign key column of the user tables in the GeoPackage.

    """
    with closing(sqlite3.connect(gpkg)) as con:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'gpkg\\_%' ESCAPE '\\' "
                "AND name NOT LIKE 'rtree\\_%' ESCAPE '\\' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
            )
        ]
        targets = []
        for t in tables:
            # fk[3] maps to foreign_key column in target table
            for fk in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                targets.append((t, fk[3]))

        total = len(targets)
        for idx, (t, c) in enumerate(targets, 1):
            name = f"idx_{t}_{c}"
            print(f"\rWriting foreign key indices: {idx}/{total}", end="", flush=True)
            con.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{t}" ("{c}")')
            con.execute("ANALYZE")
            con.commit()
        print()  # Newline at the end
