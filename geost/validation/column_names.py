from typing import Iterable

POSSIBLE_COLUMN_NAMING = {
    "nr": {"nr", "bro_id", "nitg_nr", "nitg", "boorp"},
    "surface": {"surface", "maaiveld", "mv", "height_nap", "surface_nap"},
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
            if col in POSSIBLE_COLUMN_NAMING[column_type]
        ),
        None,
    )
    return name
