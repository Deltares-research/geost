import numpy as np
import pandas as pd
import xarray as xr

from geost.base import Collection, DiscreteData, LayeredData
from geost.models import VoxelModel


def add_voxelmodel_variable(
    data: Collection, model: VoxelModel, variable: str
) -> Collection:
    var_select = model.select_with_points(data.header.gdf)[variable]
    nrs = data.header["nr"].loc[var_select["idx"]]

    _, _, dz = model.resolution

    var_df = _reduce_to_top_bottom(var_select, dz)
    var_df.rename(columns={"values": variable}, inplace=True)
    var_df["nr"] = nrs.loc[var_df["nr"]].values

    if isinstance(data.data, LayeredData):
        result = _add_to_layered(data.data, var_df)
    elif isinstance(data.data, DiscreteData):
        pass
    else:
        raise NotImplementedError(
            "Other datatypes than LayeredData or DiscreteData not implemented yet"
        )

    return result.to_collection()


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
    reduced = df.groupby(["nr", "values"])["bottom"].min().reset_index()
    return reduced


def _add_to_layered(data: LayeredData, variable: pd.DataFrame) -> LayeredData:
    """
    Helper function for `add_voxelmodel_variable` to combine a voxelmodel variable in
    layered data form. This joins the DataFrames of the LayeredData instance and the
    variable from the voxelmodel to add and updates the "top" and "bottom" columns in the
    LayeredData.

    Parameters
    ----------
    data : LayeredData
        LayeredData instance to add the voxelmodel variable to.
    variable : pd.DataFrame
        DataFrame with the voxelmodel variable to add. Contains columns ["nr", "bottom",
        variable] where variable is for example "strat".

    Returns
    -------
    LayeredData
        New instance of `LayeredData` with the added variable and corrected top and bottom
        depths.

    """
    data_df = data._change_depth_values(data.df.copy())
    data_df = pd.concat(_add_layered(data_df, variable), ignore_index=True)

    data_df["top"] = data_df["surface"] - data_df["top"]
    data_df["bottom"] = data_df["surface"] - data_df["bottom"]

    return LayeredData(data_df)


def _add_layered(data: pd.DataFrame, variable: pd.DataFrame):
    """
    Helper function to `_add_to_layered` to combine data per individual data object.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame from LayeredData instance (see docstring `_add_to_layered`).
    variable : pd.DataFrame
        DataFrame with the voxelmodel variable to add (see docstring `_add_to_layered`).

    Yields
    ------
    pd.DataFrame
        New DataFrame with the added variable for each individual data object.

    """
    data = pd.concat([data, variable], ignore_index=True)
    data.sort_values(by=["nr", "bottom"], ascending=[True, False], inplace=True)
    for _, df in data.groupby("nr"):
        df = df.bfill()
        bottom_shift_down = df["bottom"].shift()
        df.loc[df["top"] > bottom_shift_down, "top"] = bottom_shift_down
        yield df
