import numpy as np
import pandas as pd
import xarray as xr

from geost.base import Collection, DiscreteData, LayeredData
from geost.models import VoxelModel


def add_voxelmodel_variable(
    collection: Collection, model: VoxelModel, variable: str
) -> Collection:
    """
    Add a information from a variable of a `VoxelModel` instance as a column to the data
    of a `BoreholeCollection` or `CptCollection` instance. This checks for each data object
    in the Collection instance in which voxel stack the object is located and adds the
    relevant layer boundaries of the variable to the data object of the Collection based
    on depth.

    Parameters
    ----------
    collection : Collection
        Any GeoST `Collection` object such as a :class:`~geost.base.BoreholeCollection`
        or :class:`~geost.base.CptCollection` to add the `VoxelModel` variable to.
    model : :class:`~geost.modelbase.VoxelModel`
        `VoxelModel` instance to add the information from to the `Collection` instance.
    variable : str
        Name of the variable in the `VoxelModel` to add.

    Returns
    -------
    Collection
        New `Collection` (same type as the input) object with the variable added to the
        data object of the `Collection`.

    Raises
    ------
    NotImplementedError
        Raises error for geost data object types that are not yet implemented such as
        :class:`~geost.base.LineData` (future developments).

    """
    var_select = model.select_with_points(collection.header.gdf)[variable]
    nrs = collection.header["nr"].loc[var_select["idx"]]

    _, _, dz = model.resolution

    var_df = _reduce_to_top_bottom(var_select, dz)
    var_df.rename(columns={"values": variable}, inplace=True)
    var_df["nr"] = nrs.loc[var_df["nr"]].values

    if isinstance(collection.data, LayeredData):
        result = _add_to_layered(collection.data, var_df)
    elif isinstance(collection.data, DiscreteData):
        result = _add_to_discrete(collection.data, var_df)
    else:
        raise NotImplementedError(
            "Other datatypes than LayeredData or DiscreteData not implemented yet."
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
            "bottom": np.tile(da["z"] - (0.5 * dz), da.sizes["idx"]),
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
    variable = variable.merge(
        data[["nr", "surface"]].drop_duplicates(), on="nr", how="left"
    )
    variable["bottom"] = variable["surface"] - variable["bottom"]

    result = pd.concat(_add_layered(data.df, variable), ignore_index=True)
    result.dropna(subset=["top"], inplace=True)
    return LayeredData(result)


def _add_to_discrete(data: DiscreteData, variable: pd.DataFrame) -> DiscreteData:
    """
    Helper function for `add_voxelmodel_variable` to combine a voxelmodel variable in
    discrete data form. This joins the DataFrames of the DiscreteData instance and the
    variable from the voxelmodel to return a new instance of DiscreteData.

    Parameters
    ----------
    data : DiscreteData
        DiscreteData instance to add the voxelmodel variable to.
    variable : pd.DataFrame
        DataFrame with the voxelmodel variable to add. Contains columns ["nr", "bottom",
        variable] where variable is for example "strat".

    Returns
    -------
    DiscreteData
        New instance of `DiscreteData` with the added variable.

    """
    variable.rename(columns={"bottom": "depth"}, inplace=True)
    variable = variable.merge(
        data[["nr", "surface"]].drop_duplicates(), on="nr", how="left"
    )
    variable["depth"] = variable["surface"] - variable["depth"]

    result = (
        pd.concat([data.df, variable])
        .sort_values(by=["nr", "depth"])
        .reset_index(drop=True)
    )
    nr = result["nr"]
    result = pd.concat([nr, result.groupby("nr").bfill()], axis=1)
    result.dropna(subset=["end"], inplace=True)
    return DiscreteData(result)


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
    data.sort_values(by=["nr", "bottom"], inplace=True)
    for _, df in data.groupby("nr"):
        df = df.bfill()
        bottom_shift_down = df["bottom"].shift()
        df.loc[df["top"] < bottom_shift_down, "top"] = bottom_shift_down
        yield df
