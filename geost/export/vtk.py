from typing import Iterable

import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr
from pyvista import CellType, MultiBlock


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
    data_columns: list[str],
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
    data_columns: list[str],
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


def voxelmodel_to_pyvista_unstructured(
    dataset: xr.Dataset,
    resolution: tuple[float, float, float],
    displayed_variables: list[str] = None,
) -> pv.UnstructuredGrid:
    """
    Convert an xarray dataset to a PyVista voxel model (UnstructuredGrid).

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset containing 3D grid data with coordinates 'x', 'y', and 'z'.
    displayed_variables : list of str
        List of variable names in the dataset to include as cell data in the voxel model.
    resolution : tuple of float
        The resolution (dx, dy, dz) representing the size of each voxel along the x, y,
        and z axes.

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        The resulting voxel model as a PyVista UnstructuredGrid, with cell data assigned
        for each variable in `displayed_variables`.
    """
    if displayed_variables is None:
        displayed_variables = dataset.data_vars

    if "z" not in dataset.coords:
        raise ValueError(
            "Dataset must contain 'z' dimension. Use xarray.Dataset.rename() to rename "
            + "the corresponding dimension to 'z' if applicable."
        )

    # Order dataset dimensions to match required order
    expected_order = ("y", "x", "z")
    if tuple(dataset.dims.keys()) != expected_order:
        dataset = dataset.transpose(*expected_order)

    # Get half resolution (used to compute voxel corners with respect to cell centers)
    xres, yres, zres = resolution[0] / 2, resolution[1] / 2, resolution[2] / 2

    # Compute cell centers (Making use of the fact that xarray coordinates represent
    # those cell centers, so we can use them directly with meshgrid)
    xx, yy, zz = np.meshgrid(dataset.x.values, dataset.y.values, dataset.z.values)
    cell_centers = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    # Compute voxel corners by offsetting cell centers with half resolution in all
    # directions.
    offsets = np.array(
        [
            [-xres, -yres, -zres],
            [xres, -yres, -zres],
            [-xres, yres, -zres],
            [xres, yres, -zres],
            [-xres, -yres, zres],
            [xres, -yres, zres],
            [-xres, yres, zres],
            [xres, yres, zres],
        ]
    )
    voxels = cell_centers[:, None, :] + offsets[None, :, :]
    points = voxels.reshape(-1, 3)

    # Build cell connectivity
    n_voxels = cell_centers.shape[0]
    cells_voxel = np.arange(n_voxels * 8).reshape((n_voxels, 8))

    # Create unstructured grid and assign data variables
    grid = pv.UnstructuredGrid({CellType.VOXEL: cells_voxel}, points)
    for var in displayed_variables:
        if not all(dim in dataset[var].dims for dim in ["x", "y", "z"]):
            print(
                f"Variable '{var}' does not have the required dimensions 'x', 'y', and 'z'. Skipping this variable."
            )
            continue
        data = dataset[var].values
        grid.cell_data[var] = data.flatten(order="C")

    # Take the first data var to check for NaN values and extract only cells in grid
    # that are not NaN. (perhaps this should be done for all data vars?)
    nan_mask = np.isnan(grid.cell_data[grid.cell_data.keys()[0]])
    if np.any(nan_mask):
        grid = grid.extract_cells(~nan_mask)

    return grid
