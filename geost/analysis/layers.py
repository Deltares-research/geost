from functools import singledispatch

import numpy as np
import pandas as pd


def get_layer_top(
    data: pd.DataFrame,
    column: str,
    value: int | float | str | list[str] | slice,
    min_thickness: float = 0,
    min_fraction: float = 0,
) -> pd.DataFrame:
    if not data.gst.has_depth_columns:
        raise ValueError(
            "Data must contain columns specifying depth intervals. See "
            "GeostFrame.has_depth_columns for more information."
        )

    data = data.copy()
    if data.gst._top and data.gst._bottom:
        tops = _get_layer_top_layered(data, column, value, min_thickness, min_fraction)
    else:
        tops = _get_layer_top_discrete(data, column, value, min_thickness, min_fraction)

    return tops


def get_layer_base(
    data: pd.DataFrame,
    column: str,
    value: int | float | str | list[str] | slice,
    min_thickness: float = 0,
    min_fraction: float = 0,
) -> pd.DataFrame:
    if not data.gst.has_depth_columns:
        raise ValueError(
            "Data must contain columns specifying depth intervals. See "
            "GeostFrame.has_depth_columns for more information."
        )

    data = data.copy()
    pass


def _get_layer_top_layered(
    data: pd.DataFrame,
    column: str,
    value: int | float | str | list[str] | slice,
    min_thickness: float = 0,
    min_fraction: float = 0,
) -> pd.DataFrame:
    value_mask = _mask(value, data[column])

    return value_mask


def _get_layer_top_discrete(
    data: pd.DataFrame,
    column: str,
    value: int | float | str | list[str] | slice,
    min_thickness: float = 0,
    min_fraction: float = 0,
) -> pd.DataFrame:
    if "thickness" not in data.columns:
        data["thickness"] = data.gst.calculate_thickness()

    value_mask = _mask(value, data[column])

    return value_mask


@singledispatch
def _mask(values, series):
    """
    Helper function to create a Boolean Series by checking equality of values using different
    object types.

    """
    raise TypeError(
        f"Unsupported type of selection values: {type(values)}. Values must be str, list[str], or slice."
    )


@_mask.register(int | float | str)
def _(values, series):
    return series == values


@_mask.register(set | list | np.ndarray | pd.Series)
def _(values, series):
    return series.isin(values)


@_mask.register(slice)
def _(values, series):
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Can only use a slice on numerical columns.")
    start = values.start or -1e34
    stop = values.stop or 1e34
    return series.between(start, stop)


def find_top_sand(
    lith: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    min_sand_frac: float,
    min_sand_thickness: float,
) -> float:
    """
    Find the top of sand depth in a borehole described in NEN5104 format. The top of sand
    is defined by the first layer of a specified thickness that contains a minimum
    percentage of sand. By default: when the first layer of sand is detected, the next 1
    meter is scanned. Within this meter, if more than 50% of the length has a main
    lithology of sand, the initially detected layer of sand is regarded as the top
    of sand. If not, continue downward until the next layer of sand is detected and
    repeat.

    Parameters
    ----------
    lith : ndarray
        Numpy array containing the lithology of the borehole.
    top : ndarray
        Numpy array containing the top depth of the layers of the borehole.
    bottom : ndarray
        Numpy array containing the bottom depth of the layers of the borehole.
    min_sand_frac : float
        Minimum percentage required to be sand.
    min_sand_thickness : float
        Minimum thickness of the sand to search for.

    Returns
    -------
    top_sand : float
        Top depth of the sand layer that meets the requirements.

    """
    is_sand = ("Z" == lith) + ("G" == lith)

    found_sand = False
    if np.any(is_sand):
        idx_sand = np.flatnonzero(is_sand)
        for idx in idx_sand:
            top_sand = top[idx]
            search_depth = top_sand + min_sand_thickness

            search_mask = (top >= top_sand) & (top < search_depth)

            tmp_top = top[search_mask].copy()
            tmp_bottom = bottom[search_mask].copy()

            if tmp_bottom[-1] > search_depth:
                tmp_bottom[-1] = search_depth

            length = tmp_bottom - tmp_top

            sand_frac = length[is_sand[search_mask]].sum() / min_sand_thickness

            if sand_frac >= min_sand_frac:
                found_sand = True
                break

    if not found_sand:
        top_sand = np.nan

    return top_sand


def top_of_sand(
    boreholes: pd.DataFrame,
    ids: str = "nr",
    min_sand_frac: float = 0.5,
    min_sand_thickness: int | float = 1,
):
    """
    Find the top of sand depth in a borehole described in NEN5104 format. The top of sand
    is defined by the first layer of a specified thickness that contains a minimum fraction
    of sand.

    Parameters
    ----------
    boreholes : pd.DataFrame
        Boreholes in NEN5104 format with "lith", "top" and "bottom" columns.
    ids : str, optional
        Column specifying the borehole IDs. The default is "nr".
    min_sand_frac : float, optional
        Minimum percentage of sand in the sand layer. The default is 0.5 (=50%).
    min_sand_thickness : int | float, optional
        Minimum thickness of the sand layer to find the top of. The default is 1.

    Returns
    ------
    pd.DataFrame
        DataFrame containing the borehole IDs and the top of sand depths.

    """
    groupby = boreholes.groupby(ids)

    result = []
    for nr, df in groupby:
        lith = df["lith"].values
        top = df["top"].values
        bottom = df["bottom"].values

        top_sand = find_top_sand(lith, top, bottom, min_sand_frac, min_sand_thickness)

        result.append((nr, top_sand))

    return pd.DataFrame(result, columns=["nr", "top"])
