from itertools import batched
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import pyvista as pv


def _get_pyvista():  # pragma: no cover
    """Lazy import for pyvista."""
    try:
        import pyvista as pv

        return pv
    except ImportError:
        raise ImportError(
            "pyvista is required for VTK export functionality. "
            "Install it with: pip install pyvista"
        )


def prepare_as_continuous(
    data: pd.DataFrame,
    depth_column: Literal["depth", "bottom"],
    vertical_factor: float,
) -> np.ndarray:
    """
    Prepare data for PyVista cylinder objects as continuous depth representation (e.g.
    using 'depth' or 'bottom' as depth column). This is typical for CPTs and well logs.
    """
    data_xyz = data[["x", "y", depth_column]].to_numpy(copy=True)
    data_xyz[:, -1] *= vertical_factor
    return data_xyz


def prepare_as_layers(
    data: pd.DataFrame,
    depth_column: list[Literal["top"], Literal["bottom"]],
    vertical_factor: float,
) -> np.ndarray:
    """
    Prepare data for PyVista cylinder objects using layer representation (e.g. using
    'top' and 'bottom' as depth columns). This is typical for boreholes with layer
    descriptions and e.g. non-continuous layer intervals of grainsize sample data.
    """
    data_xyz_top = data[["x", "y", depth_column[0]]].to_numpy()
    data_xyz_bottom = data[["x", "y", depth_column[-1]]].to_numpy()
    data_xyz = np.column_stack([data_xyz_top, data_xyz_bottom]).reshape(-1, 3)
    data_xyz[:, -1] *= vertical_factor
    return data_xyz


def generate_cylinders(
    data: pd.DataFrame,
    depth_column: Literal["depth", "bottom"] | list[Literal["top"], Literal["bottom"]],
    data_columns: list[str],
    radius: float,
    n_sides: int,
    vertical_factor: float,
) -> Iterable:
    pv = _get_pyvista()

    boreholes = data.groupby("nr")
    for _, borehole in boreholes:
        # Case I  - depth_column is a single column representing depth from surface
        # (e.g. 'depth' or only 'bottom')
        if isinstance(depth_column, str):
            borehole_prepared = prepare_as_continuous(
                borehole, depth_column, vertical_factor
            )
            poly = pv.PolyData(borehole_prepared)
            line_segments = np.arange(0, len(borehole_prepared), dtype=np.int_)
            line_segments = np.insert(line_segments, 0, len(borehole_prepared))
            poly.lines = line_segments

            # Apply data fields
            poly.point_data.update(
                {
                    data_column: borehole[data_column].values
                    for data_column in data_columns
                }
            )
            cylinder = poly.tube(radius=radius, n_sides=n_sides)

        # Case II - depth_column is a list of two column names representing top and
        # bottom depths (e.g. ['top', 'bottom'])
        elif isinstance(depth_column, list):
            borehole_prepared = prepare_as_layers(
                borehole, depth_column, vertical_factor
            )
            cylinder = pv.merge(
                [
                    pv.Tube(
                        pointa=b[0],
                        pointb=b[1],
                        radius=radius,
                        n_sides=n_sides,
                        capping=True,
                    )
                    for b in batched(borehole_prepared, 2, strict=True)
                ],
                merge_points=True,
            )

            # Apply data fields
            cylinder.cell_data.update(
                {
                    data_column: np.repeat(
                        borehole[data_column].values, repeats=n_sides + 2
                    )
                    for data_column in data_columns
                }
            )
        yield cylinder


def borehole_to_multiblock(
    data: pd.DataFrame,
    depth_column: Literal["depth", "bottom"] | list[Literal["top"], Literal["bottom"]],
    displayed_variables: list[str],
    radius: float,
    n_sides: int,
    vertical_factor: float,
    fixed_surface: bool,
) -> pv.MultiBlock:
    """
    Create a PyVista MultiBlock object from the parsed boreholes/cpt's.

    Parameters
    ----------
    data : pd.DataFrame
        Table of borehole/CPT objects. This is CptCollection.data or
        BoreholeCollection.data.
    depth_column : Literal['depth', 'bottom'] | list[Literal["top"], Literal["bottom"]]
        Name of the column or columns representing depth.
    displayed_variables : List[str]
        Column names of data columns to write in the vtk file
    radius : float
        Radius of borehole cylinders
    n_sides : int
        Number of sides for the cylinder
    vertical_factor : float
        Vertical adjustment factor to convert e.g. heights in cm to m.

    Returns
    -------
    pv.MultiBlock
        MultiBlock object with boreholes represented as cylinder geometries

    """
    pv = _get_pyvista()

    cylinders = generate_cylinders(
        data,
        depth_column,
        displayed_variables,
        radius,
        n_sides,
        vertical_factor,
    )
    cylinders_multiblock = pv.MultiBlock(list(cylinders))
    return cylinders_multiblock


def layerdata_to_pyvista_unstructured(
    data: pd.DataFrame,
    depth_column: Literal["depth", "bottom"] | list[Literal["top"], Literal["bottom"]],
    displayed_variables: list[str],
    radius: float = 1.0,
) -> pv.UnstructuredGrid:
    """
    Convert a layerdata object to a PyVista UnstructuredGrid.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing at least columns x, y, surface, top, and bottom.
    depth_column : Literal['depth', 'bottom'] | list[Literal["top"], Literal["bottom"]]
        Name of the column or columns representing depth.
    displayed_variables : list of str
        List of variable names in the data to include as cell data in the voxel model.
    radius : float
        The 'radius' of the voxels. This will determine the
        horizontal size of the voxels in the resulting unstructured grid.

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        The resulting voxel model as a PyVista UnstructuredGrid.
    """
    pv = _get_pyvista()

    x = data["x"].values
    y = data["y"].values

    # Case I  - depth_column is a single column representing depth from surface
    # (e.g. 'depth' or only 'bottom'). In this case we compute the top depth.
    if isinstance(depth_column, str):
        top = data[depth_column].shift(1)
        top[data["nr"] != data["nr"].shift(1)] = data["surface"]
        bottom = data[depth_column].values
    # Case II - depth_column is a list of two column names representing top and
    # bottom depths (e.g. ['top', 'bottom'])
    elif isinstance(depth_column, list):
        top = data[depth_column[0]].values
        bottom = data[depth_column[-1]].values

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
    grid = pv.UnstructuredGrid({pv.CellType.VOXEL: cells_voxel}, points)
    for var in displayed_variables:
        if var not in data.columns:
            print(
                f"Variable '{var}' is unavailable in the data. Skipping this variable."
            )
            continue
        grid.cell_data[var] = data[var].values
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
    pv = _get_pyvista()

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
        if not all(dim in dataset[var].sizes for dim in {"x", "y", "z"}):
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
    pv = _get_pyvista()

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
    grid = pv.UnstructuredGrid({pv.CellType.VOXEL: cells_voxel}, points)
    for var in displayed_variables:
        if not all(dim in dataset[var].dims for dim in {"x", "y", "z"}):
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
        grid = grid.extract_cells(
            ~nan_mask
        )  # , pass_point_ids=False, pass_cell_ids=False) #TODO: add kwargs for pyvista 0.47+
        grid = grid.clean(produce_merge_map=False)

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
