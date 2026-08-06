import operator
from typing import Literal

import numpy as np
import xarray as xr


def slice_depth_interval(
    model: xr.Dataset | xr.DataArray,
    upper: int | float | np.ndarray | xr.DataArray = None,
    lower: int | float | np.ndarray | xr.DataArray = None,
    how: Literal["overlap", "majority", "inner"] = "overlap",
    drop: bool = True,
):
    """
    See docstring of :meth:`geost.models.ModelDataset.slice_depth_interval` for details.

    """
    sliced = model.copy()
    _, _, zres = model.gst.resolution()

    check_upper = operator.lt if how == "overlap" else operator.le
    check_lower = operator.gt if how == "overlap" else operator.ge

    if how == "overlap":
        upper_bound = sliced.gst.z - 0.5 * zres
        lower_bound = sliced.gst.z + 0.5 * zres
    elif how == "majority":
        upper_bound = sliced.gst.z
        lower_bound = sliced.gst.z
    elif how == "inner":
        upper_bound = sliced.gst.z + 0.5 * zres
        lower_bound = sliced.gst.z - 0.5 * zres
    else:
        raise ValueError(
            "Invalid value for 'how', use 'overlap', 'majority' or 'inner'"
        )

    if upper is not None:
        upper, upper_bound = _check_to_broadcast(upper, upper_bound, model)
        sliced = sliced.where(check_upper(upper_bound, upper), drop=drop)

    if lower is not None:
        lower, lower_bound = _check_to_broadcast(lower, lower_bound, model)
        sliced = sliced.where(check_lower(lower_bound, lower), drop=drop)

    return sliced


def _check_to_broadcast(
    values: int | float | np.ndarray | xr.DataArray,
    bounds: xr.DataArray,
    ds: xr.Dataset | xr.DataArray,
):
    """
    Helper function for `slice_depth_interval` to broadcast selection criteria if needed.

    """
    if isinstance(values, (int, float)):
        return values, bounds

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

    values, bounds, _ = xr.broadcast(values, bounds, ds)

    return values, bounds


def most_common(da: xr.DataArray, return_thickness=False) -> xr.DataArray | xr.Dataset:
    """
    See docstring of :meth:`geost.models.ModelDataArray.most_common` for details.

    """
    from scipy import stats

    x_dim, y_dim = da.gst.x_dim, da.gst.y_dim

    result, counts = stats.mode(
        da, axis=da.get_axis_num(da.gst.z_dim), nan_policy="omit"
    )
    *_, zres = da.gst.resolution()

    result = xr.DataArray(
        result, coords={y_dim: da[y_dim], x_dim: da[x_dim]}, dims=(y_dim, x_dim)
    )

    if return_thickness:
        thickness = counts * zres
        thickness = xr.DataArray(
            thickness, coords={y_dim: da[y_dim], x_dim: da[x_dim]}, dims=(y_dim, x_dim)
        )
        result = xr.Dataset({"most_common": result, "thickness": thickness})

    return result
