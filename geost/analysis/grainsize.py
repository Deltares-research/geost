from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from geost import BoreholeCollection


def __sort_gs_data(
    single_sample_data: pd.DataFrame,
) -> pd.Series:
    """
    Helper function to compute sorted percentages or masses for a single sample.

    Parameters
    ----------
    single_sample_data : pd.DataFrame
        DataFrame containing at least the columns 'd_low' (grain size lower limit),
        'd_high' (grain size upper limit) and 'percentage' (mass percentage for each
        grain size fraction) or 'mass'. This function expects a DataFrame of a single
        unique sample (unique borehole id + sample id).

    Returns
    -------
    pd.DataFrame
        Sorted cumulative percentages for the sample.
    """
    sorted_group = single_sample_data.sort_values("d_low")
    if "percentage" in sorted_group.columns:
        sorted_group["percentage_or_mass"] = sorted_group["percentage"]
    elif "mass" in sorted_group.columns:
        sorted_group["percentage_or_mass"] = sorted_group["mass"]

    return sorted_group


def calculate_grainsize_percentiles(
    single_sample_data: pd.DataFrame,
    percentiles: int | float | list[int | float] = 50,
    only_sand: bool = False,
) -> float | dict:
    """
    Calculate the grain size(s) corresponding to given cumulative mass percentile(s).
    Returns the estimated grain size(s) at the specified percentile(s) based on the input sample data.

    Parameters
    ----------
    single_sample_data : pd.DataFrame
        DataFrame containing at least the columns 'd_low' (grain size lower limit),
        'd_high' (grain size upper limit) and 'percentage' (mass percentage for each
        grain size fraction) or 'mass'. This function expects a DataFrame of a single
        unique sample (unique borehole id + sample id).
    percentiles : int, float, or list of int/float, optional
        The cumulative mass percentile(s) for which to estimate the grain size(s)
        (default is 50, the median grain size d50).
    only_sand : bool, optional
        If True, only the sand fraction is considered for the calculation (default is
        False). Note that the grain sizes must be given in micrometers (µm) for this to
        work.

    Returns
    -------
    float or dict
        Estimated grain size at the specified percentile if a single percentile is given,
        or a dict of {percentile: grainsize} if multiple percentiles are given.
    """
    if isinstance(percentiles, (int, float)):
        percentiles = [percentiles]

    if only_sand:
        single_sample_data = single_sample_data[
            (single_sample_data["d_low"] >= 62.5)
            & (single_sample_data["d_high"] <= 2000)
        ]

    if single_sample_data.empty:
        return None

    cumulative = single_sample_data["percentage_or_mass"].cumsum()
    cumulative_normalized = cumulative / cumulative.iloc[-1] * 100

    def interpolate_percentile(percentile):
        idx = np.searchsorted(cumulative_normalized, percentile)
        if idx == 0:
            return 0
        else:
            p0 = single_sample_data["percentage_or_mass"].iloc[idx - 1]
            p1 = single_sample_data["percentage_or_mass"].iloc[idx]
            y0 = single_sample_data["d_low"].iloc[idx - 1]
            y1 = single_sample_data["d_high"].iloc[idx]
            return y1 - (y1 - y0) * (p0 / (p0 + p1))

    if only_sand:
        result = {f"d{p}_sand": interpolate_percentile(p) for p in percentiles}
    else:
        result = {f"d{p}": interpolate_percentile(p) for p in percentiles}

    return result


def calculate_grainsize_fractions(
    single_sample_data: pd.DataFrame,
) -> dict:
    """
    Calculate the grain size fractions (clay, silt, sand, gravel) for a given sample.

    Parameters
    ----------
    single_sample_data : pd.DataFrame
        DataFrame containing at least the columns 'd_low' (grain size lower limit),
        'd_high' (grain size upper limit) and 'percentage' (mass percentage for each
        grain size fraction) or 'mass'. This function expects a DataFrame of a single
        unique sample (unique borehole id + sample id).
    """
    normalized = (
        single_sample_data["percentage_or_mass"]
        / single_sample_data["percentage_or_mass"].sum()
        * 100
    )

    fines = normalized.loc[
        single_sample_data[single_sample_data["d_high"] <= 62.5].index
    ].sum()
    sand = normalized.loc[
        single_sample_data[
            (single_sample_data["d_low"] >= 62.5)
            & (single_sample_data["d_high"] < 2000)
        ].index
    ].sum()
    gravel = normalized.loc[
        single_sample_data[single_sample_data["d_low"] >= 2000].index
    ].sum()

    fractions = {
        "perc_fines": fines,
        "perc_sand": sand,
        "perc_gravel": gravel,
    }

    return fractions


def calculate_grainsize_metrics(
    sample_data: pd.DataFrame,
    percentiles: int | float | list[int | float] = 50,
    only_sand: bool = False,
    fractions: bool = False,
):
    """
    Calculate the grain size metrics at specific percentiles for each sample in the
    provided DataFrame (e.g. D10, D50, D90).

    Parameters
    ----------
    sample_data : pd.DataFrame
        DataFrame containing the grain size distribution data for multiple boreholes and
        samples following the minimum requirements for a GeoST grainsize data table.
    percentiles : int, float, or list of int/float, optional
        The cumulative mass percentile(s) for which to estimate the grain size(s)
        (default is 50, the median grain size d50).
    only_sand : bool, optional
        If True, only the sand fraction is considered for the calculation (default is
        False).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the estimated grain size metrics at the specified percentiles
        for each sample.
    """
    results = []

    for _, sample_group in sample_data.groupby(["nr", "sample_nr"]):
        sample_metrics = sample_group.drop(
            ["d_low", "d_high", "percentage", "mass"], axis=1, errors="ignore"
        ).iloc[0]

        sorted_sample_group = __sort_gs_data(sample_group)
        percentile_results = calculate_grainsize_percentiles(
            sorted_sample_group, percentiles=percentiles, only_sand=only_sand
        )
        if percentile_results:
            sample_metrics = pd.concat([sample_metrics, pd.Series(percentile_results)])

            if fractions:
                fraction_results = calculate_grainsize_fractions(sorted_sample_group)
                sample_metrics = pd.concat(
                    [sample_metrics, pd.Series(fraction_results)]
                )
                results.append(sample_metrics)

    df_metrics = pd.DataFrame(results)
    return df_metrics
