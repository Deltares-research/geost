import numpy as np
import pandas as pd
import xarray as xr

from geost.base import Collection, DiscreteData, LayeredData
from geost.models import VoxelModel
from geost.models.model_utils import label_consecutive_2d


def add_voxelmodel_variable(
    collection: Collection, model: VoxelModel, variable: str
) -> Collection:
    """
    Add a information from a variable of a `VoxelModel` instance as a column to the data
    of a `BoreholeCollection` or `CptCollection` instance. This checks for each data object
    in the Collection instance in which voxel stack the object is located and adds the
    relevant layer boundaries of the variable to the data object of the Collection based
    on depth.

    Note
    ----
    If the variable name is already present in the columns of the data attribute of the
    collection, the present column is overwritten. To avoid this, rename the variable
    before either in collection.data or in the voxelmodel.

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
    if variable in collection.data.df.columns:
        collection.data.df.drop(columns=[variable], inplace=True)

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
    layer_ids = label_consecutive_2d(da.values, axis=1)
    df = pd.DataFrame(
        {
            "nr": np.repeat(da["idx"], da.sizes["z"]),
            "bottom": np.tile(da["z"] - (0.5 * dz), da.sizes["idx"]),
            "layer": layer_ids.ravel(),
            "values": da.values.ravel(),
        }
    )
    reduced = df.pivot_table(
        index=["nr", "layer", "values"], values="bottom", aggfunc="min"
    ).reset_index()
    return reduced.drop(columns=["layer"])


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

    result = (
        pd.concat([data.df, variable])
        .sort_values(by=["nr", "bottom", "top"])
        .reset_index(drop=True)
    )
    nr = result["nr"]
    result = pd.concat([nr, result.groupby("nr").bfill()], axis=1)
    result.drop_duplicates(subset=["nr", "bottom"], inplace=True)
    result = _reset_tops(result)
    result.dropna(subset=["top"], inplace=True)
    return LayeredData(result)


def _reset_tops(layered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function for `_add_to_layered` to reset the top depths after stratigraphic
    layer boundaries have been added and backfilling has been done. This changes tops into
    bottoms of the row above in each borehole if tops are lower than bottoms. It also checks
    if bottoms are less than 0 and resets these layers to 0. Then it removes layers with
    a thickness of 0.

    Parameters
    ----------
    layered_df : pd.DataFrame
        Pandas DataFrame with layered data containing tops and bottoms.

    Returns
    -------
    pd.DataFrame
        DataFrame with the tops reset.
    """
    bottom_shift_down = layered_df["bottom"].shift()
    nr_shift_down = layered_df["nr"].shift()

    layered_df.loc[
        (layered_df["top"] < bottom_shift_down) & (layered_df["nr"] == nr_shift_down),
        "top",
    ] = bottom_shift_down

    layered_df.loc[layered_df["bottom"] < 0, "bottom"] = 0

    return layered_df[layered_df["top"] - layered_df["bottom"] != 0].reset_index(
        drop=True
    )


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
    result.drop_duplicates(subset=["nr", "depth"], inplace=True)
    result.dropna(subset=["end"], inplace=True)
    return DiscreteData(result)
