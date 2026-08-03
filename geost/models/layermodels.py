import numpy as np
import xarray as xr


def slice_depth_interval(
    ds: xr.Dataset | xr.DataArray,
    upper: int | float | np.ndarray | xr.DataArray = None,
    lower: int | float | np.ndarray | xr.DataArray = None,
    update_top_bottom: bool = True,
    drop: bool = True,
):
    raise NotImplementedError(
        "The slice_depth_interval function is not implemented for layermodels. "
    )
