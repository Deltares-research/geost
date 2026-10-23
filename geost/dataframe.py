from typing import Any

import pandas as pd


def pandas_dataframe(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Convenience function to create a pandas DataFrame that is compatible with the ``.gst``
    accessor without explicitly having to ``import pandas``. The DataFrame you want to
    create should at least have a survey ID column, otherwise a `MissingSurveyIDError` is
    raised because the result would not be compatible with the ``.gst`` accessor.

    Parameters
    ----------
    *args, **kwargs : Any
        Any positional or keyword arguments passed to ``pd.DataFrame``, see the
        Pandas documentation for more details.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame that is compatible with the ``.gst`` accessor.

    Examples
    --------
    >>> import geost
    >>> df = geost.pandas_dataframe(
    ...     {
    ...         "nr": ["A", "A", "B", "B"],
    ...         "top": [0, 0.5, 0, 0.5],
    ...         "bottom": [0.5, 1, 0.5, 1],
    ...         "lith": ["sand", "clay", "sand", "clay"],
    ...     }
    ... )
    >>> df
      nr  top  bottom  lith
    0  A  0.0     0.5  sand
    1  A  0.5     1.0  clay
    2  B  0.0     0.5  sand
    3  B  0.5     1.0  clay

    """
    dataframe = pd.DataFrame(*args, **kwargs)
    dataframe.gst._nr  # Raises a MissingSurveyIDError if the survey ID column is not present
    return dataframe
