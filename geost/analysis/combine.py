from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geost.base import Collection
from geost.models.model_utils import label_consecutive_2d

if TYPE_CHECKING:
    import xarray as xr

    from geost.models import VoxelModel


def add_voxelmodel_variable(
    collection: Collection, model: VoxelModel, variable: str
) -> Collection:
    """
    Add a information from a variable of a `VoxelModel` instance as a column to the data
    of a :class:`~geost.base.Collection` instance. This checks for each survey in the
    Collection in which voxel stack the survey is located and adds the relevant layer
    boundaries of the variable to the data object of the Collection based on depth.

    Note
    ----
    If the variable name is already present in the columns of the data attribute of the
    collection, the present column is overwritten. To avoid this, rename the variable
    before either in collection.data or in the voxelmodel.

    Parameters
    ----------
    collection : :class:`~geost.base.Collection`
        `Collection` to add the `VoxelModel` variable to.
    model : :class:`~geost.models.VoxelModel`
        `VoxelModel` instance to add the information from to the `Collection` instance.
    variable : str
        Name of the variable in the `VoxelModel` to add.

    Returns
    -------
    :class:`~geost.base.Collection`
        `Collection` instance with the information from the `VoxelModel` variable added
        to the data table.

    """
    data = collection.data

    if variable in data.columns:
        data.drop(columns=[variable], inplace=True)

    columns: dict = data.gst.positional_columns
    nr_, surface_, top_, bottom_ = (
        columns["nr"],
        columns["surface"],
        columns["top"],
        columns["depth"],
    )

    surface_level = data.groupby(nr_).agg({surface_: "first"})

    var_select = model.select_with_points(collection.header)
    var_select = var_select.assign_coords(
        idx=collection.header[nr_].loc[var_select["idx"]]
    )

    _, _, dz = model.resolution

    var_df = _reduce_to_top_bottom(
        var_select, dz, survey_name=nr_, depth_name=bottom_, value_name=variable
    )
    var_df = var_df.merge(surface_level, left_on=nr_, right_index=True, how="left")
    var_df = var_df[
        var_df[bottom_] < var_df[surface_]
    ]  # Only keep layers below surface, strat boundaries are bottoms of layers
    var_df[bottom_] = var_df[surface_] - var_df[bottom_]
    var_df = var_df.drop(columns=surface_)

    result = pd.concat([data, var_df]).sort_values(by=[nr_, bottom_])
    nr = result[nr_]  # Keep survey ID because it is lost in the groupby backfill step
    result = pd.concat([nr, result.groupby(nr_).bfill()], axis=1)
    result = (
        result.dropna(subset=surface_)
        .drop_duplicates(subset=[nr_, bottom_])
        .reset_index(drop=True)
    )
    # Drop duplicate bottom values because combining may cause duplicate depths

    if top_ is not None:
        result = _reset_tops(result, nr=nr_, top=top_, bottom=bottom_)

    return Collection(
        result,
        header=collection.header.copy(),
        has_inclined=collection.has_inclined,
        vertical_reference=collection.vertical_reference,
    )


def _reduce_to_top_bottom(
    da: xr.DataArray,
    dz: int | float,
    survey_name: str = "idx",
    depth_name: str = "bottom",
    value_name: str = "values",
) -> pd.DataFrame:
    """
    Helper to reduce the selection DataArray from `VoxelModel.select_with_points` to a
    DataFrame containing "idx", "top" and "bottoms" of relevant layer boundaries.

    Parameters
    ----------
    da : xr.DataArray
        Selection result of `VoxelModel.select_with_points`.
    dz : int | float
        Vertical size of voxels in the VoxelModel.
    depth_name : str
        Name to use for the depth column in the resulting DataFrame.

    Returns
    -------
    pd.DataFrame

    """
    da = da[value_name]
    layer_ids = label_consecutive_2d(da.values, axis=1)
    df = pd.DataFrame(
        {
            survey_name: np.repeat(da["idx"], da.sizes["z"]),
            depth_name: np.tile(da["z"] - (0.5 * dz), da.sizes["idx"]),
            "layer": layer_ids.ravel(),
            value_name: da.values.ravel(),
        }
    )
    reduced = df.pivot_table(
        index=[survey_name, "layer", value_name], values=depth_name, aggfunc="min"
    ).reset_index()
    return reduced.drop(columns=["layer"])


def _reset_tops(layered: pd.DataFrame, nr: str, top: str, bottom: str) -> pd.DataFrame:
    """
    Helper function for `add_voxelmodel_variable` to reset the top depths of layered data
    after stratigraphic layer boundaries have been added and backfilling has been done.
    This changes tops into bottoms of the row above in each borehole if tops are lower
    than bottoms and removes any layers with a thickness of 0 in the result.

    Parameters
    ----------
    layered : pd.DataFrame
        Pandas DataFrame with layered data containing tops and bottoms.
    nr : str
        Column name for the survey ID.
    top : str
        Column name for the top depths.
    bottom : str
        Column name for the bottom depths.

    Returns
    -------
    pd.DataFrame
        DataFrame with the tops reset.

    """
    bottom_shift_down = layered[bottom].shift()
    nr_shift_down = layered[nr].shift()

    layered.loc[
        (layered[top] < bottom_shift_down) & (layered[nr] == nr_shift_down),
        top,
    ] = bottom_shift_down

    # Remove layers with zero thickness
    layered = layered[layered[top] - layered[bottom] != 0].reset_index(drop=True)

    return layered
