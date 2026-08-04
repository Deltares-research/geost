import numpy as np
import xarray as xr


def slice_depth_interval(
    model: xr.Dataset | xr.DataArray,
    upper: int | float | np.ndarray | xr.DataArray = None,
    lower: int | float | np.ndarray | xr.DataArray = None,
    update_top_bottom: bool = True,
    drop: bool = True,
):
    """
    See docstring of :meth:`geost.models.ModelDataset.slice_depth_interval` for details.

    """
    _top = model.gst._top
    _bottom = model.gst._bottom
    z_dim = model.gst.z_dim

    if isinstance(model, xr.Dataset):
        # We only slice variables which include the z dimension, so we need to separate them
        # from the other variables in the dataset
        vars_3d = [var for var in model.data_vars if z_dim in model[var].dims]
        other_vars = [var for var in model.data_vars if var not in vars_3d]
        sliced = model[vars_3d].copy()
    else:
        sliced = model.copy()

    if upper is not None:
        upper_bound = _check_to_broadcast(upper, sliced)
        sliced = sliced.where(sliced[_bottom] <= upper_bound, drop=drop)

    if lower is not None:
        lower_bound = _check_to_broadcast(lower, sliced)
        sliced = sliced.where(sliced[_top] >= lower_bound, drop=drop)

    if update_top_bottom:
        upper = _check_to_broadcast(upper, sliced)
        lower = _check_to_broadcast(lower, sliced)
        sliced[_top] = xr.where(sliced[_top] > upper, upper, sliced[_top])
        sliced[_bottom] = xr.where(sliced[_bottom] < lower, lower, sliced[_bottom])

    if isinstance(model, xr.Dataset):
        sliced.update(model[other_vars])

    return sliced


def _check_to_broadcast(
    values: int | float | np.ndarray | xr.DataArray,
    ds: xr.Dataset | xr.DataArray,
):
    """
    Helper function for `slice_depth_interval` to broadcast selection criteria if needed.

    """
    if isinstance(values, (int, float)):
        return values

    if isinstance(values, np.ndarray):
        y_dim, x_dim = ds.gst.y_dim, ds.gst.x_dim
        try:
            values = xr.DataArray(
                values, coords={y_dim: ds[y_dim], x_dim: ds[x_dim]}, dims=(y_dim, x_dim)
            )
        except ValueError as e:
            raise ValueError(
                "Failed to broadcast input array to dataset dimensions"
            ) from e

    values, _ = xr.broadcast(values, ds)

    return values
