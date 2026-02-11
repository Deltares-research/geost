from functools import singledispatch

import pandas as pd
from numpy.core.multiarray import interp as compiled_interp

from geost import BoreholeCollection

BHRGT_CLASSIFICATION = {
    "fractionSmaller63um": 31.5,
    "fraction63to90um": 76.5,
    "fraction90to125um": 107.5,
    "fraction125to180um": 152.5,
    "fraction180to250um": 215.0,
    "fraction250to355um": 302.5,
    "fraction355to500um": 427.5,
    "fraction500to710um": 605.0,
    "fraction710to1000um": 855.0,
    "fraction1000to1400um": 1200.0,
    "fraction1400umto2mm": 1700.0,
    "fraction2to4mm": 3000.0,
    "fraction4to8mm": 6000.0,
    "fraction8to16mm": 12000.0,
    "fraction16to31_5mm": 24250.0,
    "fraction31_5to63mm": 47500.0,
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

    Returns
    -------
    pd.DataFrame
        DataFrame containing the estimated grain size percentiles (D10, D50, D90) for each sample.
    """
    sample_data_result = sample_data.copy()

    converted_sample_data = sample_data_result[BHRGT_CLASSIFICATION.keys()].rename(
        columns=BHRGT_CLASSIFICATION
    )

    if only_sand:
        converted_sample_data.drop(
            [31.5, 3000.0, 6000.0, 12000.0, 24250.0, 47500.0], axis=1, inplace=True
        )

    cumsum_sample_data = converted_sample_data.cumsum(axis=1)
    normalized_cumsum = (
        cumsum_sample_data.div(cumsum_sample_data.iloc[:, -1], axis=0) * 100
    )

    for percentile in percentiles:
        sample_data_result[f"d{percentile}"] = [
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
