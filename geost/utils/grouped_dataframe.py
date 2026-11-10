import numpy as np
import pandas as pd


def weighted_mean(weights: np.ndarray):
    """
    _summary_

    Parameters
    ----------
    weights : np.ndarray
        _description_

    """

    def average(values: pd.Series):
        w = weights[values.index]
        is_nan = np.isnan(values)
        return np.average(values[~is_nan], weights=w[~is_nan])

    return average
