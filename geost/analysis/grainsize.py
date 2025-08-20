import numpy as np
import pandas as pd


def calculate_percentile_grainsize(
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
    # Sort by grain size lower limit
    sorted_group = single_sample_data.sort_values("d_low")
    if "percentage" in sorted_group.columns:
        sorted_group["percentage_or_mass"] = sorted_group["percentage"]
    elif "mass" in sorted_group.columns:
        sorted_group["percentage_or_mass"] = sorted_group["mass"]

    if only_sand:
        sorted_group = sorted_group[
            (sorted_group["d_low"] >= 62.5) & (sorted_group["d_high"] <= 2000)
        ]

    cumulative = sorted_group["percentage_or_mass"].cumsum()
    cumulative_normalized = cumulative / cumulative.iloc[-1] * 100

    def interpolate_percentile(percentile):
        idx = np.searchsorted(cumulative_normalized, percentile)
        if idx == 0:
            return 0
        else:
            p0 = sorted_group["percentage_or_mass"].iloc[idx - 1]
            p1 = sorted_group["percentage_or_mass"].iloc[idx]
            y0 = sorted_group["d_low"].iloc[idx - 1]
            y1 = sorted_group["d_high"].iloc[idx]
            return y1 - (y1 - y0) * (p0 / (p0 + p1))

    if isinstance(percentiles, (int, float)):
        return interpolate_percentile(percentiles)
    else:
        return {p: interpolate_percentile(p) for p in percentiles}


def calculate_dnumber(
    sample_data: pd.DataFrame,
    percentiles: int | float | list[int | float] = 50,
    only_sand=False,
):
    """
    Calculate the number of grains at a specific percentile for each sample in the
    provided DataFrame.

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
        DataFrame containing the estimated number of grains at the specified percentile
        for each sample.
    """
    # Apply the grain size calculation to each sample
    df_dnumbers = (
        sample_data.groupby(["nr", "sample_nr"])
        .apply(
            lambda sample: calculate_percentile_grainsize(
                sample, percentiles=percentiles, only_sand=only_sand
            )
        )
        .reset_index(level=[0, 1])
    )

    return df_dnumbers
