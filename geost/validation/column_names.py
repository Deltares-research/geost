from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable

from geost.config import load_user_positional_column_aliases

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_POSITIONAL_COLUMNS = {
    "nr": {"nr", "bro_id", "nitg_nr", "nitg", "boorp"},
    "surface": {"surface", "maaiveld", "mv", "height_nap", "surface_nap"},
    "end": {"end", "einddiepte", "einddiepte_nap", "end_depth", "end_depth_nap"},
    "x_coordinate": {
        "x",
        "x-coord",
        "longitude",
        "lon",
        "easting",
        "x_bottom_rd",
        "x_rd_crd",
        "x_calc_crd",
    },
    "y_coordinate": {
        "y",
        "y-coord",
        "latitude",
        "lat",
        "northing",
        "y_bottom_rd",
        "y_rd_crd",
        "y_calc_crd",
    },
    "top": {"top", "tv_top_nap", "top_diepte", "top_depth", "upperboundary"},
    "depth": {
        "depth",
        "bottom",
        "tv_bottom_nap",
        "basis_diepte",
        "bottom_depth",
        "lowerboundary",
    },
}

POSITIONAL_COLUMN_NAMES = {
    key: set(values) for key, values in DEFAULT_POSITIONAL_COLUMNS.items()
}
for key, values in load_user_positional_column_aliases().items():
    POSITIONAL_COLUMN_NAMES[key].update(map(str.lower, values))


def check_column_name(columns: Iterable[str], column_type: str) -> str | None:
    """
    Check if a column name is valid by comparing it against a set of valid names.

    Parameters
    ----------
    columns : Iterable[str]
        The incoming column names to check.
    column_type : str
        The type of column to check against.

    Returns
    -------
    str | None
        The valid column name if found, otherwise None.

    """
    valid_names = POSITIONAL_COLUMN_NAMES[column_type]
    for col in columns:
        if col.lower() in valid_names:
            return col
    return None


def check_positional_column_presence(df: pd.DataFrame) -> None:
    try:
        df.gst._nr  # Raises a KeyError if no survey ID can be found in the accessor
    except KeyError as e:
        raise KeyError(
            "Input table is missing a mandatory column that identifies survey IDs. "
            f"This can be one of: {sorted(POSITIONAL_COLUMN_NAMES['nr'])}. Use the "
            "'column_mapper' argument to specify which column identifies the survey ID."
        ) from e

    warnings_list: list[str] = []

    if not df.gst.has_xy_columns:
        warnings_list.append(
            "Input table is missing x/y coordinate columns. Accepted column names:\n"
            f"x: {sorted(POSITIONAL_COLUMN_NAMES['x_coordinate'])}\n"
            f"y: {sorted(POSITIONAL_COLUMN_NAMES['y_coordinate'])}\n"
            "Use the 'column_mapper' argument to map your columns to 'x' and 'y'."
        )

    if not df.gst.has_depth_columns:
        warnings_list.append(
            "Input table is missing depth information on surface level and top and/or "
            f"bottom depth. Accepted column names:\n"
            f"surface: {sorted(POSITIONAL_COLUMN_NAMES['surface'])}\n"
            f"top: {sorted(POSITIONAL_COLUMN_NAMES['top'])} (optional for layered data)\n"
            f"depth: {sorted(POSITIONAL_COLUMN_NAMES['depth'])} (always required)\n"
            "Use the 'column_mapper' argument to map your columns to the expected depth "
            "columns."
        )

    if warnings_list:
        warnings.warn(
            f"Issues found while checking positional columns:\n- {'\n- '.join(warnings_list)}",
            stacklevel=2,
        )
