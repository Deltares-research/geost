import json

import pytest

from geost import add_positional_columns
from geost.config import (
    delete_user_positional_column_aliases,
    load_user_positional_column_aliases,
    save_user_positional_column_aliases,
)
from geost.validation.column_names import (
    DEFAULT_COLUMN_NAMING,
    POSSIBLE_COLUMN_NAMING,
)


@pytest.fixture
def isolated_alias_config(tmp_path, monkeypatch):
    """Use an isolated config file and clean runtime registry per test."""
    import geost.config as config

    config_dir = tmp_path / "geost"
    config_file = config_dir / "column_aliases.json"

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)

    # reset runtime registry before test
    delete_user_positional_column_aliases(persist=False)

    yield config_file

    # reset again after test to avoid leakage
    delete_user_positional_column_aliases(persist=False)


@pytest.mark.unittest
def test_save_and_load_user_aliases(isolated_alias_config):
    aliases = {"x_coordinate": ["my_x", "custom_x"]}

    save_user_positional_column_aliases(aliases)
    loaded = load_user_positional_column_aliases()

    assert loaded == aliases


@pytest.mark.unittest
def test_load_missing_config_returns_empty_dict(isolated_alias_config):
    assert load_user_positional_column_aliases() == {}


@pytest.mark.unittest
def test_delete_persisted_alias_file(isolated_alias_config):
    save_user_positional_column_aliases({"x_coordinate": ["my_x"]})
    assert isolated_alias_config.exists()

    delete_user_positional_column_aliases(persist=True)

    assert not isolated_alias_config.exists()


@pytest.mark.unittest
def test_add_runtime_alias_single_value(isolated_alias_config):
    add_positional_columns({"x_coordinate": "my_x"})

    assert "my_x" in POSSIBLE_COLUMN_NAMING["x_coordinate"]


@pytest.mark.unittest
def test_add_runtime_alias_multiple_values(isolated_alias_config):
    add_positional_columns({"x_coordinate": ["my_x", "custom_x"]})

    assert "my_x" in POSSIBLE_COLUMN_NAMING["x_coordinate"]
    assert "custom_x" in POSSIBLE_COLUMN_NAMING["x_coordinate"]


@pytest.mark.unittest
def test_add_aliases_are_lowercased(isolated_alias_config):
    add_positional_columns({"x_coordinate": "MY_X"})

    assert "my_x" in POSSIBLE_COLUMN_NAMING["x_coordinate"]
    assert "MY_X" not in POSSIBLE_COLUMN_NAMING["x_coordinate"]


@pytest.mark.unittest
def test_add_persistent_alias_written_to_config(isolated_alias_config):
    add_positional_columns(
        {"x_coordinate": ["MY_X", "CUSTOM_X"]},
        persist=True,
    )

    loaded = load_user_positional_column_aliases()

    assert loaded["x_coordinate"] == ["custom_x", "my_x"]


@pytest.mark.unittest
def test_add_invalid_key_raises_value_error(isolated_alias_config):
    with pytest.raises(ValueError, match="Invalid names"):
        add_positional_columns({"invalid": "foo"})


@pytest.mark.unittest
def test_delete_resets_runtime_to_defaults(isolated_alias_config):
    add_positional_columns({"x_coordinate": "my_x"})
    assert "my_x" in POSSIBLE_COLUMN_NAMING["x_coordinate"]

    delete_user_positional_column_aliases()

    assert "my_x" not in POSSIBLE_COLUMN_NAMING["x_coordinate"]
    assert "x" in POSSIBLE_COLUMN_NAMING["x_coordinate"]


@pytest.mark.unittest
def test_delete_does_not_mutate_defaults(isolated_alias_config):
    delete_user_positional_column_aliases()

    POSSIBLE_COLUMN_NAMING["x_coordinate"].add("temporary")

    assert "temporary" not in DEFAULT_COLUMN_NAMING["x_coordinate"]
