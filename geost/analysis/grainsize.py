from functools import singledispatch

import pandas as pd
from numpy._core.multiarray import interp as compiled_interp

from geost import BoreholeCollection

BHRGT_CLASSIFICATION = {
    "fractionSmaller63um": 63,
    "fraction63to90um": 90,
    "fraction90to125um": 125,
    "fraction125to180um": 180,
    "fraction180to250um": 250,
    "fraction250to355um": 355,
    "fraction355to500um": 500,
    "fraction500to710um": 710,
    "fraction710to1000um": 1000,
    "fraction1000to1400um": 1400,
    "fraction1400umto2mm": 2000,
    "fraction2to4mm": 4000,
    "fraction4to8mm": 8000,
    "fraction8to16mm": 16000,
    "fraction16to31_5mm": 31500,
    "fraction31_5to63mm": 63000,
}


@singledispatch
def calculate_bhrgt_grainsize_percentiles(
    sample_data: pd.DataFrame | BoreholeCollection,
    percentiles: int | float | list[int | float] = 50,
    only_sand: bool = False,
) -> pd.DataFrame | BoreholeCollection:
    """
    Calculate the grain size percentiles (e.g. D10, D50, D90) for each sample in the
    provided DataFrame.

    Parameters
    ----------
    sample_data : pd.DataFrame | BoreholeCollection
        DataFrame or BoreholeCollection containing the grain size distribution data for
        multiple boreholes and samples following the minimum requirements for a
        GeoST-BHRGT-samples grainsize data table.

    Returns
    -------
    pd.DataFrame | BoreholeCollection
        DataFrame or BoreholeCollection containing the estimated grain size percentiles
        (D10, D50, D90) for each sample.
    """
    raise NotImplementedError(
        f"Grain size percentile calculation not implemented for type {type(sample_data)}"
    )


@calculate_bhrgt_grainsize_percentiles.register
def _(
    sample_data: pd.DataFrame,
    percentiles: int | float | list[int | float] = 50,
    only_sand: bool = False,
) -> pd.DataFrame:
    """
    Calculate the grain size percentiles (e.g. D10, D50, D90) for each sample in the
    provided DataFrame.

    Parameters
    ----------
    sample_data : pd.DataFrame
        DataFrame containing the grain size distribution data for multiple boreholes and
        samples following the minimum requirements for a GeoST-BHRGT-samples grainsize
        data table.
    percentiles : int | float | list[int | float], optional
        Percentiles to calculate (e.g. 10, 50, 90). The default is 50 (D50).
    only_sand : bool, optional
        If True, only calculate percentiles based on the sand fractions (i.e. ignore
        fractions smaller than 63um and larger than 2mm). The default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the estimated grain size percentiles for each sample. The
        percentile columns are named "d{percentile}" (e.g. "d50") or "d{percentile}_sand"
        (e.g. "d50_sand") if only_sand is True.
    """
    sample_data_result = sample_data.copy()

    if isinstance(percentiles, (int, float)):
        percentiles = [percentiles]

    converted_sample_data = sample_data_result[BHRGT_CLASSIFICATION.keys()].rename(
        columns=BHRGT_CLASSIFICATION
    )

    if only_sand:
        converted_sample_data.drop(
            [63, 4000, 8000, 16000, 31500, 63000], axis=1, inplace=True
        )
        converted_sample_data.insert(0, 63, 0)
    else:
        converted_sample_data.insert(0, 0, 0)

    cumsum_sample_data = converted_sample_data.cumsum(axis=1)
    normalized_cumsum = (
        cumsum_sample_data.div(cumsum_sample_data.iloc[:, -1], axis=0) * 100
    )

    for percentile in percentiles:
        percentile_col_name = f"d{percentile}_sand" if only_sand else f"d{percentile}"
        sample_data_result[percentile_col_name] = [
            compiled_interp(
                percentile, normalized_cumsum.values[i], normalized_cumsum.columns
            )
            for i in range(normalized_cumsum.values.shape[0])
        ]

    return sample_data_result


@calculate_bhrgt_grainsize_percentiles.register
def _(
    sample_data: BoreholeCollection,
    percentiles: int | float | list[int | float] = 50,
    only_sand: bool = False,
) -> BoreholeCollection:
    sample_data.data = calculate_bhrgt_grainsize_percentiles(
        sample_data.data, percentiles=percentiles, only_sand=only_sand
    )
    return sample_data


@singledispatch
def calculate_bhrgt_grainsize_fractions(
    sample_data: pd.DataFrame | BoreholeCollection,
) -> pd.DataFrame:
    """
    Calculate fractions of fines, sand and gravel for each sample in the provided
    DataFrame.

    Parameters
    ----------
    sample_data : pd.DataFrame | BoreholeCollection
        DataFrame or BoreholeCollection containing the grain size distribution data for
        multiple boreholes and samples following the minimum requirements for a
        GeoST-BHRGT-samples grainsize data table.

    Returns
    -------
    pd.DataFrame | BoreholeCollection
        DataFrame or BoreholeCollection containing the estimated grain size fractions
        (fines, sand, gravel) for each sample.
    """
    raise NotImplementedError(
        f"Grain size fraction calculation not implemented for type {type(sample_data)}"
    )


@calculate_bhrgt_grainsize_fractions.register
def _(
    sample_data: pd.DataFrame,
) -> pd.DataFrame:
    sample_data["perc_fines"] = sample_data["fractionSmaller63um"]
    sample_data["perc_sand"] = sample_data[
        [
            "fraction63to90um",
            "fraction90to125um",
            "fraction125to180um",
            "fraction180to250um",
            "fraction250to355um",
            "fraction355to500um",
            "fraction500to710um",
            "fraction710to1000um",
            "fraction1000to1400um",
            "fraction1400umto2mm",
        ]
    ].sum(axis=1)
    sample_data["perc_gravel"] = sample_data[
        [
            "fraction2to4mm",
            "fraction4to8mm",
            "fraction8to16mm",
            "fraction16to31_5mm",
            "fraction31_5to63mm",
            "fractionLarger63mm",
        ]
    ].sum(axis=1)

    return sample_data


@calculate_bhrgt_grainsize_fractions.register
def _(
    sample_data: BoreholeCollection,
) -> BoreholeCollection:
    sample_data.data = calculate_bhrgt_grainsize_fractions(sample_data.data)
    return sample_data
