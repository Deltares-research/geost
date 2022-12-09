import numpy as np
import pyvista as pv
import pandas as pd
from typing import Iterable, List


def __prepare_borehole(borehole: pd.DataFrame, vertical_factor) -> np.ndarray:
    bh_as_pnts = borehole[["x", "y", "bottom"]].to_numpy().astype(np.float64)
    bh = np.vstack(
        [
            np.array([bh_as_pnts[0, 0], bh_as_pnts[0, 1], borehole["mv"].iloc[0]]),
            bh_as_pnts,
        ]
    )
    bh[:, 2] *= vertical_factor
    return bh


def __create_tubes(
    table: pd.DataFrame,
    data_columns: List[str],
    radius: float,
    vertical_factor: float,
) -> Iterable:
    boreholes = table.groupby("nr")
    for _, borehole in boreholes:
        borehole_prepared = __prepare_borehole(borehole, vertical_factor)
        poly = pv.PolyData(borehole_prepared)
        line_segments = np.arange(0, len(borehole_prepared), dtype=np.int_)
        line_segments = np.insert(line_segments, 0, len(borehole_prepared))
        poly.lines = line_segments

        for data_column in data_columns:
            poly[data_column] = np.hstack(
                [borehole[data_column].values[0], borehole[data_column].values]
            )
        yield poly.tube(radius=radius)


def borehole_to_multiblock(
    table: pd.DataFrame,
    data_columns: List[str],
    radius,
    vertical_factor,
) -> pv.MultiBlock:
    """
    Create a PyVista MultiBlock object from the parsed boreholes/cpt's.

    Parameters
    ----------
    table : pd.DataFrame
        Table of borehole/CPT objects. This is CptCollection.data or BoreholeCollection.data
    data_columns : List[str]
        Column names of data arrays to write in the vtk file
    radius : float
        Radius of borehole tubes
    vertical_factor : float
        Vertical adjustment factor to convert e.g. heights in cm to m.

    Returns
    -------
    pv.MultiBlock
        MultiBlock object with borehole geometries
    """

    tubes = __create_tubes(table, data_columns, radius, vertical_factor)
    return pv.MultiBlock(list(tubes))
