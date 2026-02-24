from typing import Iterable

possible_column_naming = {
    "x_coordinate": set(["x", "x-coord", "longitude", "lon", "easting", "x_bottom_rd"]),
    "y_coordinate": set(["y", "y-coord", "latitude", "lat", "northing", "y_bottom_rd"]),
    "top": set(["top", "tv_top_nap"]),
    "depth": set(["depth", "bottom", "tv_bottom_nap"]),
}


def check_column_name(columns: Iterable[str], column_type: str) -> str:
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
    str
        The valid column name if found, otherwise None.
    """
    name = next(
        (
            col
            for col in columns.str.lower()
            if col in possible_column_naming[column_type]
        ),
        None,
    )
    return name
