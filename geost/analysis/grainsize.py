import numpy as np
import pandas as pd


def calculate_percentile_grainsize(
    single_sample_data: pd.DataFrame,
    percentile: int | float = 50,
    only_sand: bool = False,
) -> float:
    """
    Calculate the grain size corresponding to a given cumulative mass percentile. E.g.
    when you want to calculate median grain size (percentile=50) or d90 (percentile=90),
    this function returns the estimated grain size at the specified percentile based on
    the input sample data.

    Parameters
    ----------
    single_sample_data : pd.DataFrame
        DataFrame containing at least the columns 'd_low' (grain size lower limit),
        'd_high' (grain size upper limit) and 'percentage' (mass percentage for each
        grain size fraction). This function expects a DataFrame of a single unique
        sample (unique borehole id + sample id).
    percentile : int or float, optional
        The cumulative mass percentile for which to estimate the grain size (default is
        50, the median grain size d50).
    only_sand : bool, optional
        If True, only the sand fraction is considered for the calculation (default is
        False). Note that the grain sizes must be given in micrometers (µm) for this to
        work.

    Returns
    -------
    float
        Estimated grain size at the specified percentile.
    """
    # Sort by grain size lower limit
    sorted_group = single_sample_data.sort_values("d_low")

    if only_sand:
        sorted_group = sorted_group[
            (sorted_group["d_low"] >= 62.5) & (sorted_group["d_high"] <= 2000)
        ]

    # Calculate cumulative sum of mass percentages and normalize in case cum_mass[-1] != 100
    # This can be the case if only considering sand fraction or if the data is incomplete (#TODO: cum percentage = 100 validation)
    cum_mass = sorted_group["percentage"].cumsum()
    cum_mass_norm = cum_mass / cum_mass.iloc[-1] * 100

    # Find closest index
    idx = np.searchsorted(cum_mass_norm, percentile)
    if idx == 0:
        return 0
    else:
        # Interpolate to estimate the grainsize at given percentile
        x0 = cum_mass_norm.iloc[idx - 1]
        x1 = cum_mass_norm.iloc[idx]
        y0 = sorted_group["d_low"].iloc[idx - 1]
        y1 = sorted_group["d_high"].iloc[idx - 1]
        return y0 + (percentile - x0) * (y1 - y0) / (x1 - x0)


def calculate_dnumber(
    sample_data: pd.DataFrame, percentile: int | float = 50, only_sand=False
):
    """
    Calculate the number of grains at a specific percentile for each sample in the
    provided DataFrame.

    Parameters
    ----------
    sample_data : pd.DataFrame
        DataFrame containing the grain size distribution data for multiple boreholes and
        samples following the minimum requirements for a GeoST grainsize data table.
    percentile : int or float, optional
        The cumulative mass percentile for which to estimate the number of grains
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
        (
            sample_data.groupby(["nr", "sample_nr"]).apply(
                lambda sample: calculate_percentile_grainsize(
                    sample, percentile=percentile, only_sand=only_sand
                )
            )
        )
        .reset_index(level=[0, 1])
        .rename(columns={0: f"d{percentile}"})
    )

    return df_dnumbers
