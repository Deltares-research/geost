from enum import Enum
from typing import NamedTuple

import xarray as xr

from geost.exceptions import InvalidModelError

XCOORD_NAMES = {"x", "xco", "xcoord", "longitude", "lon", "easting"}
YCOORD_NAMES = {"y", "yco", "ycoord", "latitude", "lat", "northing"}
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


class ModelSpec(NamedTuple):
    x_dim: str
    y_dim: str
    z_dim: str
    model_type: ModelType
    top: str | None = None
    bottom: str | None = None


def get_model_specs(xarray_obj: xr.DataArray | xr.Dataset) -> ModelSpec:
    """
    Detect the horizontal and vertical dimension information for layer- and voxelmodel
    datasets. In voxelmodels, the vertical dimension contains the depth information of
    each voxel. In layermodels, the vertical dimension contains the unit and depth information
    is stored in "top" and "bottom" like data variables. In case of layermodels, the top
    and bottoms are also detected.

    Parameters
    ----------
    xarray_obj : xr.Dataset | xr.DataArray
        Voxelmodel or layermodel dataset or dataarray.

    Returns
    -------
    ModelSpec
        X,Y, and Vertical dimension information for the model containing the name of the
        z-dimension and the inferred model type. In case of layermodels, the names of the
        top and bottom data variables are also returned.

    Raises
    ------
    ValueError
        If both a voxelmodel and layermodel vertical dimension is detected, an error is
        raised.

    """
    x_dim = xarray_obj.rio._x_dim  # Utilize rioxarray if possible
    y_dim = xarray_obj.rio._y_dim

    if x_dim is None:
        for coord in xarray_obj.coords:
            if coord.lower() in XCOORD_NAMES:
                x_dim = coord
                break

    if y_dim is None:
        for coord in xarray_obj.coords:
            if coord.lower() in YCOORD_NAMES:
                y_dim = coord
                break

    # Find the vertical dimension: "z" or "layer" like names
    dims = tuple(xarray_obj.dims)

    voxel_match = None
    layer_match = None
    for d in dims:
        if d.lower() in VOXELMODEL_Z_NAMES:
            voxel_match = d
        if d.lower() in LAYERMODEL_Z_NAMES:
            layer_match = d

    if voxel_match and layer_match:
        raise InvalidModelError(
            f"Ambiguous vertical dimension: voxel={voxel_match}, layer={layer_match}"
        )

    if voxel_match:
        return ModelSpec(
            x_dim=x_dim, y_dim=y_dim, z_dim=voxel_match, model_type=ModelType.VOXEL
        )

    if layer_match:
        top, bottom = detect_top_and_bottom(xarray_obj)
        return ModelSpec(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=layer_match,
            model_type=ModelType.LAYER,
            top=top,
            bottom=bottom,
        )


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
                return top, bottom

    for coord in ds.coords:
        if coord.lower() in LAYERMODEL_TOP_NAMES:
            top = coord

        if coord.lower() in LAYERMODEL_BOTTOM_NAMES:
            bottom = coord

        if top and bottom:
            break

    return top, bottom
