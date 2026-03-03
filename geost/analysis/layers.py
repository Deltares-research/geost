import numpy as np
import pandas as pd

from geost.utils import series


def get_layer_top(
    data: pd.DataFrame,
    column: str,
    value: int | float | str | list[str] | slice,
    min_thickness: float = None,
    min_fraction: float = None,
) -> pd.DataFrame:
    if not data.gst.has_depth_columns:
        raise ValueError(
            "Data must contain columns specifying depth intervals. See "
            "GeostFrame.has_depth_columns for more information."
        )

    data = data.copy()
    data["values_mask"] = series.mask(value, data[column])

    if data.gst._top is None:
        data["thickness"] = data.gst.calculate_thickness()
        data["top"] = data[data.gst._bottom] - data["thickness"]

    if min_thickness is not None:
        if min_thickness <= 0:
            raise ValueError("min_thickness must be positive and zero or greater.")

        data["layer_nrs"] = series.label_consecutive_elements(data["values_mask"])

        data = _get_layer_top_bottom(data)
        data["thickness"] = data.gst.calculate_thickness()

    if min_fraction is not None and not (0 <= min_fraction <= 1):
        raise ValueError("min_fraction must be between 0 and 1.")

    tops = _get_layer_top(data, min_thickness, min_fraction)
    return tops


def _get_layer_top_bottom(data: pd.DataFrame) -> pd.DataFrame:
    top_col = data.gst._top
    bottom_col = data.gst._bottom

    top_bottom = data.groupby(["nr", "layer_nrs"], as_index=False).agg(
        {"surface": "first", top_col: "min", bottom_col: "max", "values_mask": "first"}
    )

    return top_bottom


def get_layer_base(
    data: pd.DataFrame,
    column: str,
    value: int | float | str | list[str] | slice,
    min_thickness: float = None,
    min_fraction: float = None,
) -> pd.DataFrame:
    if not data.gst.has_depth_columns:
        raise ValueError(
            "Data must contain columns specifying depth intervals. See "
            "GeostFrame.has_depth_columns for more information."
        )

    data = data.copy()
    pass


def _get_layer_top(
    data: pd.DataFrame,
    min_thickness: float = None,
    min_fraction: float = None,
) -> pd.DataFrame:
    top_col = data.gst._top

    if min_thickness is not None:
        if min_fraction is not None:
            return _find_top(data.groupby("nr"), min_thickness, min_fraction)

        selection = data[data["values_mask"] & (data["thickness"] >= min_thickness)]
    else:
        selection = data[data["values_mask"]]

    tops = selection.groupby("nr", as_index=False)[top_col].min()

    return tops


def _find_top(grouped, min_thickness, min_fraction):
    pass


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


def cumulative_thickness():
    pass
