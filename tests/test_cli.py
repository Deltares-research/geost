import pytest
from typer.testing import CliRunner

from geost.cli import app
from geost.download import pdok_geopackage

runner = CliRunner()

COMMANDS = {
    "download-bhrgt-geopackage": "BHR-GT",
    "download-bhrg-geopackage": "BHR-G",
    "download-bhrp-geopackage": "BHR-P",
    "download-cpt-geopackage": "CPT",
    "download-sfr-geopackage": "SFR",
}
KEEP_ZIP_COMMANDS = [c for c in COMMANDS if c != "download-bhrg-geopackage"]
FK_COMMANDS = [c for c in COMMANDS if c != "download-sfr-geopackage"]


@pytest.fixture
def calls(monkeypatch, tmp_path):
    recorded = {"download": [], "create_bro_index": [], "write_fk_index": []}

    def fake_download(type_, savedir, **kwargs):
        recorded["download"].append((type_, savedir, kwargs))
        return tmp_path / "fake.gpkg"

    monkeypatch.setattr(pdok_geopackage, "download", fake_download)
    monkeypatch.setattr(
        pdok_geopackage,
        "create_bro_index",
        lambda *args: recorded["create_bro_index"].append(args),
    )
    monkeypatch.setattr(
        pdok_geopackage,
        "write_fk_index",
        lambda *args: recorded["write_fk_index"].append(args),
    )
    return recorded


@pytest.mark.unittest
def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.unittest
@pytest.mark.parametrize("command, type_", COMMANDS.items())
def test_command_runs(calls, command, type_):
    result = runner.invoke(app, [command, "outdir"])

    assert result.exit_code == 0
    assert f"Saved {type_} geopackage in outdir" in result.output
    assert len(calls["download"]) == 1
    assert calls["download"][0][:2] == (type_, "outdir")
    assert len(calls["create_bro_index"]) == (0 if type_ == "SFR" else 1)
    assert calls["write_fk_index"] == []


@pytest.mark.unittest
@pytest.mark.parametrize("command", COMMANDS)
def test_timeout_and_chunk_size_forwarded(calls, command):
    result = runner.invoke(
        app, [command, "outdir", "--timeout", "5", "--chunk-size", "1024"]
    )

    assert result.exit_code == 0
    kwargs = calls["download"][0][2]
    assert kwargs["timeout"] == 5
    assert kwargs["chunk_size_mb"] == 1024


@pytest.mark.unittest
@pytest.mark.parametrize("command", KEEP_ZIP_COMMANDS)
def test_keep_zip_file(calls, command):
    runner.invoke(app, [command, "outdir"])
    runner.invoke(app, [command, "outdir", "--keep-zip-file"])

    assert calls["download"][0][2]["delete_zip"] is True
    assert calls["download"][1][2]["delete_zip"] is False


@pytest.mark.unittest
@pytest.mark.parametrize("command", FK_COMMANDS)
def test_write_foreign_key_index(calls, command):
    result = runner.invoke(app, [command, "outdir", "--write-foreign-key-index"])

    assert result.exit_code == 0
    assert len(calls["write_fk_index"]) == 1
