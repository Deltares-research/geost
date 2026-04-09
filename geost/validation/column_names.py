import warnings
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd

POSSIBLE_COLUMN_NAMING = {
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
    valid_names = POSSIBLE_COLUMN_NAMING[column_type]
    for col in map(str.lower, columns):
        if col in valid_names:
            return col
    return None


def check_positional_column_presence(df: pd.DataFrame) -> None:
    try:
        df.gst._nr  # Raises a KeyError if no survey ID can be found in the accessor
    except KeyError as e:
        raise KeyError(
            "Input table is missing a mandatory column that identifies survey IDs. "
            f"This can be one of: {sorted(POSSIBLE_COLUMN_NAMING['nr'])}. Use the "
            "'column_mapper' argument to specify which column identifies the survey ID."
        ) from e

    warnings_list: list[str] = []

    if not df.gst.has_xy_columns:
        warnings_list.append(
            "Input table is missing x/y coordinate columns. Accepted column names:\n"
            f"x: {sorted(POSSIBLE_COLUMN_NAMING['x_coordinate'])}\n"
            f"y: {sorted(POSSIBLE_COLUMN_NAMING['y_coordinate'])}\n"
            "Use the 'column_mapper' argument to map your columns to 'x' and 'y'."
        )

    if not df.gst.has_depth_columns:
        warnings_list.append(
            "Input table is missing depth information on surface level and top and/or "
            f"bottom depth. Accepted column names:\n"
            f"surface: {sorted(POSSIBLE_COLUMN_NAMING['surface'])}\n"
            f"top: {sorted(POSSIBLE_COLUMN_NAMING['top'])} (optional for layered data)\n"
            f"depth: {sorted(POSSIBLE_COLUMN_NAMING['depth'])} (always required)\n"
            "Use the 'column_mapper' argument to map your columns to the expected depth "
            "columns."
        )

    if warnings_list:
        warnings.warn(
            f"Issues found while checking positional columns:\n- {'\n- '.join(warnings_list)}",
            stacklevel=2,
        )
