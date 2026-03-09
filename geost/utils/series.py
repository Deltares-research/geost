from functools import singledispatch

import numpy as np
import pandas as pd


def label_consecutive_elements(s: pd.Series) -> pd.Series:
    """
    Label consecutive identical elements in a pandas Series with unique group numbers.

    Parameters
    ----------
    s : pd.Series
        The input pandas Series containing elements to be labeled.

    Returns
    -------
    pd.Series
        A pandas Series with the same index as the input, where each element is replaced
        by a group number indicating its consecutive group.

    Example
    -------
    >>> s = pd.Series(['A', 'A', 'B', 'B', 'B', 'A', 'A', 'C'])
    >>> label_consecutive_elements(s)
    0    1
    1    1
    2    2
    3    2
    4    2
    5    3
    6    3
    7    4
    dtype: int64

    """
    return (s != s.shift()).cumsum()


@singledispatch
def mask(values: int | float | str | list[str] | slice, series: pd.Series) -> pd.Series:
    """
    Get a Boolean mask from a Pandas Series based on the type of selection values. In case
    of a slice, the mask will be True for values between the start and stop of the slice. In
    case of other types, the mask will be True for values equal to the selection values.

    Parameters
    ----------
    values : int | float | str | list[str] | slice
        The selection values to create the mask from. Can be a single value, a list of values,
        or a slice.
    series : pd.Series
        The Pandas Series to create the mask from.

    Returns
    -------
    pd.Series
        A Boolean mask indicating which values in the series match the selection criteria.

    Raises
    ------
    TypeError
        If the type of selection values is not supported.

    Example
    -------
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> mask(3, s)
    0    False
    1    False
    2     True
    3    False
    4    False
    dtype: bool

    >>> mask([2, 4], s)
    0    False
    1     True
    2    False
    3     True
    4    False
    dtype: bool

    >>> mask(slice(2, 4), s)
    0    False
    1     True
    2     True
    3     True
    4    False
    dtype: bool

    """
    raise TypeError(
        f"Unsupported type of selection values: {type(values)}. Values must be str, list[str], or slice."
    )


@mask.register(int | float | str)
def _(values, series):
    return series == values


@mask.register(set | list | np.ndarray | pd.Series)
def _(values, series):
    return series.isin(values)


@mask.register(slice)
def _(values, series):
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Can only use a slice on numerical columns.")
    start = values.start or -1e34
    stop = values.stop or 1e34
    return series.between(start, stop)
