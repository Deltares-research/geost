from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geost.base import Collection
from geost.models.model_utils import label_consecutive_2d

if TYPE_CHECKING:
    import xarray as xr

    from geost.accessors.data import DiscreteData, LayeredData
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
    data = collection.data

    if variable in data.columns:
        data.drop(columns=[variable], inplace=True)

    columns = data.gst.positional_columns
    nr_, surface_, top_, bottom_ = (
        columns["nr"],
        columns["surface"],
        columns["top"],
        columns["depth"],
    )

    var_select = model.select_with_points(collection.header)[variable]
    nrs = collection.header[nr_].loc[var_select["idx"]]

    _, _, dz = model.resolution

    var_df = _reduce_to_top_bottom(var_select, dz)
    var_df[nr_] = nrs.loc[var_df[nr_]].values
    var_df.rename(columns={"bottom": bottom_, "values": variable}, inplace=True)
    var_df = var_df.merge(data[[nr_, surface_]].drop_duplicates(), on=nr_, how="left")
    var_df[bottom_] = var_df[surface_] - var_df[bottom_]

    result = (
        pd.concat([data, var_df]).sort_values(by=[nr_, bottom_]).reset_index(drop=True)
    )
    nr = result[nr_]
    result = pd.concat([nr, result.groupby(nr_).bfill()], axis=1)
    result.drop_duplicates(subset=[nr_, bottom_], inplace=True)

    if top_ is not None:
        result = _reset_tops(result, nr=nr_, top=top_, bottom=bottom_)

    return Collection(
        result,
        header=collection.header.copy(),
        has_inclined=collection.has_inclined,
        vertical_reference=collection.vertical_reference,
    )


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


def _reset_tops(
    layered_df: pd.DataFrame, nr: str, top: str, bottom: str
) -> pd.DataFrame:
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
    bottom_shift_down = layered_df[bottom].shift()
    nr_shift_down = layered_df[nr].shift()

    layered_df.loc[
        (layered_df[top] < bottom_shift_down) & (layered_df[nr] == nr_shift_down),
        top,
    ] = bottom_shift_down

    layered_df.loc[layered_df[bottom] < 0, bottom] = 0

    return layered_df[layered_df[top] - layered_df[bottom] != 0].reset_index(drop=True)
