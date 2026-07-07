import pandas as pd


def reset_tops(layered: pd.DataFrame, nr: str, top: str, bottom: str) -> pd.DataFrame:
    """
    Reset the top depths of layered data when the layers have been inserted or split which
    made the depth inconsistent (i.e. top is below bottom). This changes tops into bottoms
    of the row above in each borehole if tops are lower than bottoms and removes any layers
    with a thickness of 0 in the result.

    Parameters
    ----------
    layered : pd.DataFrame
        Pandas DataFrame with layered data containing tops and bottoms.
    nr : str
        Column name for the survey ID.
    top : str
        Column name for the top depths.
    bottom : str
        Column name for the bottom depths.

    Returns
    -------
    pd.DataFrame
        DataFrame with the tops reset.

    """
    bottom_shift_down = layered[bottom].shift()
    nr_shift_down = layered[nr].shift()

    layered.loc[
        (layered[top] < bottom_shift_down) & (layered[nr] == nr_shift_down),
        top,
    ] = bottom_shift_down

    # Remove layers with zero thickness
    layered = layered[layered[top] - layered[bottom] != 0].reset_index(drop=True)

    return layered
