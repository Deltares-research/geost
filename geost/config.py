from __future__ import annotations

import json
from pathlib import Path

from platformdirs import user_config_dir

CONFIG_DIR = Path(user_config_dir("geost"))
CONFIG_FILE = CONFIG_DIR / "column_aliases.json"


class ValidationSettings:
    """
    Settings for validation behavior.

    SKIP: If True, validation will be skipped entirely.
    VERBOSE: If True, validation errors will be printed to the console.
    DROP_INVALID: If True, invalid rows will automatically be dropped from (Geo)DataFrames.
    FLAG_INVALID: If True, invalid rows will be flagged in (Geo)DataFrames. Note: only works if DROP_INVALID is False.
    AUTO_ALIGN: If True, collection headers and data tables will automatically be aligned.
    """

    SKIP = False
    VERBOSE = True
    DROP_INVALID = True
    FLAG_INVALID = False
    AUTO_ALIGN = True

    def reset_settings(self):
        """
        Reset all settings to their default values.

        """
        self.SKIP = False
        self.VERBOSE = True
        self.DROP_INVALID = True
        self.FLAG_INVALID = False
        self.AUTO_ALIGN = True


def load_user_positional_column_aliases() -> dict[str, list[str]]:
    if not CONFIG_FILE.exists():
        return {}

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_user_positional_column_aliases(aliases: dict[str, list[str]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(aliases, f, indent=2)


def delete_user_positional_column_aliases(persist: bool = False) -> None:
    """
    Reset positional column aliases to the built-in defaults.

    Parameters
    ----------
    persist : bool, optional
        If True, also remove the persisted user alias configuration file.
        The default is False.

    """
    from geost.validation.column_names import (
        DEFAULT_COLUMN_NAMING,
        POSSIBLE_COLUMN_NAMING,
    )

    POSSIBLE_COLUMN_NAMING.clear()
    for key, values in DEFAULT_COLUMN_NAMING.items():
        POSSIBLE_COLUMN_NAMING[key] = set(values)

    if persist and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


validation = ValidationSettings()
