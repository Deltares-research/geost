from functools import singledispatch
from typing import Any


def _normalize_aliases(value: str | list[str]) -> list[str]:
    values = [value] if isinstance(value, str) else value
    return [v.lower() for v in values]


def add_positional_columns(
    columns: dict[str, str | list[str]], persist: bool = False
) -> None:
    """
    Provide additional positional columns for GeoST to get methods to work correctly
    with unknown names for positional columns.

    Optional positional columns to provide are:
    - "nr": Specifies identification name/number/code of the point survey.
    - "x": Specifies X-, Easting- or lon-coordinates.
    - "y": Specifies Y-, Northing- or lat-coordinates.
    - "surface": Specifies surface elevation of surveys.
    - "end": Specifies end elevation of surveys.
    - "depth": Specifies the depth of a measurement or bottom depth of a layer.
    - "top": Specifies the top depth of a layer.

    Parameters
    ----------
    columns : dict[str, str | list[str]]
        The original columns dictionary to which positional columns will be added.
    persist : bool, optional
        If True, the changes will be saved to a user configuration file. Then additional
        columns will be loaded from this file in future sessions. The default is False.

    Returns
    -------
    None
        Adds the provided columns to the set of internal valid names for positional columns.

    Examples
    --------
    Add a single alias for the "x" positional column and only for the current session:
    >>> import geost
    >>> geost.add_positional_columns({"x_coordinate": "my-x-column"})

    Add multiple aliases for the "y" positional column and persist the changes for future
    sessions:
    >>> import geost
    >>> geost.add_positional_columns({"y_coordinate": ["my-y-column", "y_coord"]}, persist=True)

    """
    from geost.config import (
        load_user_positional_column_aliases,
        save_user_positional_column_aliases,
    )
    from geost.validation.column_names import POSSIBLE_COLUMN_NAMING

    invalid_keys = [key for key in columns if key not in POSSIBLE_COLUMN_NAMING]

    if invalid_keys:
        raise ValueError(
            f"Invalid names {', '.join(invalid_keys)} for positional columns to set. "
            f"Valid names are: {', '.join(POSSIBLE_COLUMN_NAMING.keys())}."
        )

    for key, value in columns.items():
        POSSIBLE_COLUMN_NAMING[key].update(_normalize_aliases(value))

    if persist:
        existing = load_user_positional_column_aliases()

        for key, value in columns.items():
            values = _normalize_aliases(value)
            existing.setdefault(key, [])
            existing[key] = sorted(set(existing[key]) | set(values))

        save_user_positional_column_aliases(existing)


@singledispatch
def column_name_from(values: Any, prefix: str = "", suffix: str = "") -> str:
    """
    Utility function to create a clean column name from input values for a variety of
    types.

    Parameters
    ----------
    values : Any
        Input values to create the column name from.
    prefix : str, optional
        Optional prefix to add to the column name. The default is "".
    suffix : str, optional
        Optional suffix to add to the column name. The default is "".

    Returns
    -------
    str
        The generated column name.

    Examples
    --------
    >>> column_name_from([1, 2, 3], suffix="_sum")
    '1,2,3_sum'

    >>> column_name_from("Z", prefix="mean_", suffix="_thickness")
    'mean_Z_thickness'

    >>> column_name_from(slice(1, 10), prefix="range_")
    'range_1:10'

    Raises
    ------
    TypeError
        If the input values type is not supported.

    """
    raise TypeError(f"Unsupported type: {type(values)}")


@column_name_from.register(list)
def _(values: list, prefix: str = "", suffix: str = "") -> str:
    if len(values) == 1:
        return f"{prefix}{values[0]}{suffix}"
    else:
        return f"{prefix}{','.join(map(str, values))}{suffix}"


@column_name_from.register(str)
def _(values: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{values}{suffix}"


@column_name_from.register(set)
def _(values: set, prefix: str = "", suffix: str = "") -> str:
    sorted_values = list(sorted(values))
    return column_name_from(sorted_values, prefix=prefix, suffix=suffix)


@column_name_from.register(slice)
def _(values: slice, prefix: str = "", suffix: str = "") -> str:
    start = values.start if values.start is not None else ""
    stop = values.stop if values.stop is not None else ""
    return f"{prefix}{start}:{stop}{suffix}"
