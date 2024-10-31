import numpy as np
import pandas as pd
import xarray as xr

from geost.base import Collection, LayeredData
from geost.models import VoxelModel


def add_voxelmodel_variable(
    data: Collection, model: VoxelModel, variable: str
) -> Collection:
    var_select = model.select_with_points(data.header.gdf)[variable]
    nrs = data.header["nr"].loc[var_select["idx"]]

    _, _, dz = model.resolution

    var_df = _reduce_to_top_bottom(var_select, dz)
    var_df['nr'] = nrs.loc[var_df['nr']].values
    data_df = data.data._change_depth_values(data.data.df)

    return var_df


def _reduce_to_top_bottom(da: xr.DataArray, dz: int | float) -> pd.DataFrame:
    """
    Helper to reduce the selection DataArray from `VoxelModel.select_with_points` to a
    DataFrame containing "idx", "top" and "bottoms" of relevant layer boundaries.

    Parameters
    ----------
    da : xr.DataArray
        Selection result of `VoxelModel.select_with_points`.
    dz : int | float
        Vertical size of voxels in the VoxelModel.

    Returns
    -------
    pd.DataFrame

    """
    df = pd.DataFrame(
        {
            "nr": np.repeat(da["idx"], da.sizes["z"]),
            "bottom": np.tile(da["z"] - (0.5 * dz), da.sizes["z"]),
            "values": da.values.ravel(),
        }
    )
    reduced = df.groupby(["nr", "values"])["bottom"].max().reset_index()
    return reduced
