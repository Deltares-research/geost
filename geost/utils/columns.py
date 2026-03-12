from functools import singledispatch
from typing import Any


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
