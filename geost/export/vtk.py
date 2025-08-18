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


def layerdata_to_pyvista_unstructured(
    layerdata: pd.DataFrame,
    displayed_variables: list[str],
    radius: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    Convert a layerdata object to a PyVista UnstructuredGrid.

    Parameters
    ----------
    layerdata : LayerData
        The input layerdata object containing 3D grid data with coordinates 'x', 'y',
        and 'z'.

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        The resulting voxel model as a PyVista UnstructuredGrid.
    """
    # Get the data from the layerdata object
    x = layerdata["x"].values
    y = layerdata["y"].values
    top = layerdata["surface"].values - layerdata["top"].values
    bottom = layerdata["surface"].values - layerdata["bottom"].values

    # Define all cell corner coordinates in the required order
    voxels = np.array(
        [
            [x - radius, y - radius, bottom],
            [x + radius, y - radius, bottom],
            [x - radius, y + radius, bottom],
            [x + radius, y + radius, bottom],
            [x - radius, y - radius, top],
            [x + radius, y - radius, top],
            [x - radius, y + radius, top],
            [x + radius, y + radius, top],
        ]
    )
    voxels = np.rollaxis(voxels, -1, 0)
    points = voxels.reshape(-1, 3)

    # Create the cells array
    n_voxels = len(x)
    cells_voxel = np.arange(n_voxels * 8).reshape((n_voxels, 8))

    # Create unstructured grid and assign data variables
    grid = pv.UnstructuredGrid({CellType.VOXEL: cells_voxel}, points)
    for var in displayed_variables:
        if var not in layerdata.columns:
            print(
                f"Variable '{var}' is unavailable in the layerdata. Skipping this variable."
            )
            continue
        data = layerdata[var].values
        grid.cell_data[var] = data.flatten(order="C")

    return grid


def voxelmodel_to_pyvista_structured(
    dataset: xr.Dataset,
    resolution: tuple[float, float, float],
    displayed_variables: list[str] = None,
) -> pv.StructuredGrid:
    """
    Convert an xarray dataset to a PyVista voxel model (StructuredGrid).

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset containing 3D grid data with coordinates 'x', 'y', and 'z'.
    displayed_variables : list of str
        List of variable names in the dataset to include as cell data in the voxel model.
    resolution : tuple of float
        The resolution (dy, dx, dz) representing the size of each voxel along the y, x,
        and z axes.

    Returns
    -------
    grid : pyvista.StructuredGrid
        The resulting voxel model as a PyVista StructuredGrid, with cell data assigned
        for each variable in `displayed_variables`.
    """
    if displayed_variables is None:
        displayed_variables = dataset.data_vars

    # Check if the dataset has the required dimensions and order. Because we treat the
    # voxelmodel as image data, we need dimensions in the order (x, y, z) with
    # coordinates in increasing order with respect to the origin.
    dataset = check_voxelmodel_dims(dataset, dim_order=("x", "y", "z"))

    # Define PyVista structured grid using ImageData
    grid = pv.ImageData()
    grid.spacing = resolution[1], resolution[0], resolution[2]
    grid.origin = dataset.x.values[0], dataset.y.values[0], dataset.z.values[0]
    grid.dimensions = (
        np.array(
            [dataset.sizes["x"], dataset.sizes["y"], dataset.sizes["z"]], dtype=int
        )
        + 1
    )

    # Add the variables to the grid
    for var in displayed_variables:
        if not all(dim in dataset[var].sizes for dim in ["x", "y", "z"]):
            print(
                f"Variable '{var}' does not have the required dimensions 'x', 'y', and "
                "'z'. Skipping this variable."
            )
            continue
        data = dataset[var].values
        grid.cell_data[var] = data.flatten(order="F")

    return grid


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

    # Check if the dataset has the required dimensions and order
    dataset = check_voxelmodel_dims(dataset, dim_order=("y", "x", "z"))

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
                f"Variable '{var}' does not have the required dimensions 'x', 'y', and "
                "'z'. Skipping this variable."
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


def check_voxelmodel_dims(
    dataset: xr.Dataset, dim_order: tuple = ("x", "y", "z")
) -> xr.Dataset:
    """
    Check if the voxelmodel dataset dimensions are in the required order and contain
    the required dimensions 'x', 'y', and 'z'.

    - If dimensions are missing or wrongly named, raise a ValueError.
    - If the dimensions are in the wrong order, reorder them to dim_order.
    - If the coordinates are not in increasing order, sort them.

    Unless a ValueError was raised, this will return the dataset with the correct
    dimensions. If the dataset is already in the correct order, it will be returned
    unchanged.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset to check.
    dim_order : tuple of str, optional
        The required order of dimensions. Default is ('x', 'y', 'z').

    Returns
    -------
    xr.Dataset
        The dataset with the correct dimensions and order.

    Raises
    ------
    ValueError
        If the dataset does not contain the required dimensions 'x', 'y', and 'z'.
    """
    for dim in dim_order:
        if dim not in dataset.dims or dim not in dataset.coords:
            raise ValueError(
                (
                    f"Dataset must contain '{dim}' dimension. Make sure that this "
                    "spatial dimension exists in the dataset or if it has a different "
                    "name use xarray.Dataset.rename() to rename the corresponding "
                    f"dimension to '{dim}'."
                )
            )

    # Order dataset dimensions to match required order.
    if tuple(dataset.sizes.keys()) != dim_order:
        dataset = dataset.transpose(*dim_order)
    # Also transpose all data_vars to match the required order. You have to do this
    # because the above dataset.transpose() call will not transpose the data_vars.
    for var in dataset.data_vars:
        if tuple(dataset[var].sizes) != dim_order and all(
            [dim in dataset[var].sizes for dim in dim_order]
        ):
            dataset[var] = dataset[var].transpose(*dim_order)

    # Ensure all coordinates are increasing and sort the data if needed.
    for dim in dim_order:
        coord = dataset.coords[dim]
        if coord[0] > coord[-1]:
            dataset = dataset.sortby(dim)

    return dataset
