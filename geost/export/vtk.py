from typing import Iterable, List, TypeVar

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import MultiBlock


def prepare_borehole(borehole: pd.DataFrame, vertical_factor: float) -> np.ndarray:
    borehole_xyz = borehole[["x", "y", "bottom"]].to_numpy()
    surface_xyz = np.array(
        [borehole_xyz[0, 0], borehole_xyz[0, 1], borehole["surface"].iloc[0]]
    )
    borehole_xyz = np.vstack([surface_xyz, borehole_xyz], dtype="float64")
    borehole_xyz[:, 2] *= vertical_factor
    return borehole_xyz


def generate_cylinders(
    table: pd.DataFrame,
    data_columns: List[str],
    radius: float,
    vertical_factor: float,
) -> Iterable:
    boreholes = table.groupby("nr")
    for _, borehole in boreholes:
        borehole_prepared = prepare_borehole(borehole, vertical_factor)
        poly = pv.PolyData(borehole_prepared)
        line_segments = np.arange(0, len(borehole_prepared), dtype=np.int_)
        line_segments = np.insert(line_segments, 0, len(borehole_prepared))
        poly.lines = line_segments

        for data_column in data_columns:
            poly[data_column] = np.hstack(
                [borehole[data_column].values[0], borehole[data_column].values]
            )
        cylinder = poly.tube(radius=radius)
        yield cylinder


def borehole_to_multiblock(
    table: pd.DataFrame,
    data_columns: List[str],
    radius: float,
    vertical_factor: float,
) -> MultiBlock:
    """
    Create a PyVista MultiBlock object from the parsed boreholes/cpt's.

    Parameters
    ----------
    table : pd.DataFrame
        Table of borehole/CPT objects. This is CptCollection.data or
        BoreholeCollection.data.
    data_columns : List[str]
        Column names of data arrays to write in the vtk file
    radius : float
        Radius of borehole cylinders
    vertical_factor : float
        Vertical adjustment factor to convert e.g. heights in cm to m.

    Returns
    -------
    pv.MultiBlock
        MultiBlock object with boreholes represented as cylinder geometries

    """
    cylinders = generate_cylinders(table, data_columns, radius, vertical_factor)
    cylinders_multiblock = pv.MultiBlock(list(cylinders))
    return cylinders_multiblock
