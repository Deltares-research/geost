from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import xarray as xr

VOXELMODEL_Z_NAMES = {"z", "depth", "elevation"}
LAYERMODEL_Z_NAMES = {"layer", "unit", "horizon", "stratunit"}
LAYERMODEL_TOP_NAMES = {"top", "tv_top_nap", "top_diepte", "top_depth", "upperboundary"}
LAYERMODEL_BOTTOM_NAMES = {
    "bottom",
    "tv_bottom_nap",
    "basis_diepte",
    "bottom_depth",
    "lowerboundary",
}


class ModelType(Enum):  # pragma: no cover
    VOXEL = "voxel"
    LAYER = "layer"


class VerticalSpec(NamedTuple):
    z_dim: str
    model_type: ModelType


def detect_vertical_dim(ds: xr.DataArray | xr.Dataset) -> VerticalSpec:
    """
    Detect the vertical dimension information for layer- and voxelmodel datasets. In
    voxelmodels, the vertical dimension contains the depth information of each voxel.
    In layermodels, the vertical dimension contains the unit and depth information is
    stored in "top" and "bottom" like data variables. In case of layermodels, the top
    and bottoms are also detected.

    Parameters
    ----------
    ds : xr.Dataset
        Voxelmodel or layermodel dataset.

    Returns
    -------
    VerticalSpec
        Vertical dimension information for the model containing the name of the z-dimension
        and the inferred model type.

    Raises
    ------
    ValueError
        If both a voxelmodel and layermodel vertical dimension is detected, an error is
        raised.

    """
    dims = tuple(ds.dims)

    voxel_match = None
    layer_match = None
    for d in dims:
        if d.lower() in VOXELMODEL_Z_NAMES:
            voxel_match = d
        if d.lower() in LAYERMODEL_Z_NAMES:
            layer_match = d

    if voxel_match and layer_match:
        raise ValueError(
            f"Ambiguous vertical dimension: voxel={voxel_match}, layer={layer_match}"
        )

    if voxel_match:
        return VerticalSpec(z_dim=voxel_match, model_type=ModelType.VOXEL)

    if layer_match:
        return VerticalSpec(z_dim=layer_match, model_type=ModelType.LAYER)


def detect_top_and_bottom(
    ds: xr.DataArray | xr.Dataset,
) -> tuple[str | None, str | None]:
    """
    Detect the names of top and bottom data variables in layermodel Datasets. Will only
    try to detect in `xarray.Dataset` instances as an `xarray.DataArray` cannot contain
    both variables.

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset
        Layermodel Dataset to detect the names in.

    Returns
    -------
    tuple[str | None, str | None]
        Tuple containing the names of the top and bottom data variables. If not found,
        returns None for that variable.

    """
    top = None
    bottom = None

    if isinstance(ds, xr.Dataset):
        for var_ in ds.data_vars:
            if var_.lower() in LAYERMODEL_TOP_NAMES:
                top = var_

            if var_.lower() in LAYERMODEL_BOTTOM_NAMES:
                bottom = var_

            if top and bottom:
                break

    elif isinstance(ds, xr.DataArray):
        if ds.name.lower() in LAYERMODEL_TOP_NAMES:
            top = ds.name

        if ds.name.lower() in LAYERMODEL_BOTTOM_NAMES:
            bottom = ds.name

    return top, bottom
