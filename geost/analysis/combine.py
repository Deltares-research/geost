from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from geost.base import Collection
from geost.models.model_utils import label_consecutive_2d
from geost.utils.depth import reset_tops


def add_nearest_voxelmodel_variable(
    collection: Collection,
    model: xr.Dataset | xr.DataArray,
    data_vars: str | list[str],
    tolerances: tuple[float, float, float] | None = None,
) -> Collection:
    """
    Add information from a `VoxelModel` instance as columns to the data of a
    :class:`~geost.base.Collection` instance. This checks for each survey in the
    Collection in which voxel stack the survey is located and adds the relevant data
    variables of the `VoxelModel` to the data object of the Collection based on depth.

    Note
    ----
    This function simply assigns the nearest voxel value to each layer or measurement, while
    `add_voxelmodel_variable` also updates layer boundaries based on the voxel model.

    If the variable name is already present in the columns of the data attribute of the
    collection, the present column is overwritten. To avoid this, rename the variable
    before either in collection.data or in the voxelmodel.

    Parameters
    ----------
    collection : :class:`~geost.base.Collection`
        `Collection` to add the `VoxelModel` variable to.
    model : xr.Dataset | xr.DataArray
        Model data to add the information from to the `Collection` instance.
    data_vars : str | list[str]
        Name(s) of the variable(s) in the model to add.
    tolerances : tuple[float, float, float] | None
        Optional tolerances in x, y and z direction for matching the survey points to the
        voxel centers. If not given, the x/y/z resolution of the model is used as
        tolerance.

    Returns
    -------
    :class:`~geost.base.Collection`
        `Collection` instance with the information from the model variable added.
    """

    collection_data = collection.data.gst._get_depth_relative_to_surface().copy()
    x_queried = collection_data[collection_data.gst._x].values
    y_queried = collection_data[collection_data.gst._y].values

    # Take center of layer if layered, otherwise take bottom depth as z coordinate for
    # querying the voxel model
    if collection_data.gst.is_layered:
        z_queried = (
            collection_data[collection_data.gst._top].values
            + collection_data[collection_data.gst._bottom].values
        ) / 2
    else:
        z_queried = collection_data[collection_data.gst._bottom].values

    x_, y_, z_ = model.gst.x_dim, model.gst.y_dim, model.gst.z_dim
    result = model.sel(
        {
            x_: xr.DataArray(x_queried, dims="points"),
            y_: xr.DataArray(y_queried, dims="points"),
            z_: xr.DataArray(z_queried, dims="points"),
        },
        method="nearest",
    )

    # Custom implementation of tolerance per dimension because the tolerance kwarg in
    # xarray's sel is applied to all dimensions.)
    dx = np.abs(result[x_].values - x_queried)
    dy = np.abs(result[y_].values - y_queried)
    dz = np.abs(result[z_].values - z_queried)

    # Use given tolerances or default to voxel resolution if not given
    if tolerances is not None:
        tol_x, tol_y, tol_z = tolerances
    else:
        tol_x, tol_y, tol_z = model.gst.resolution()
    mask = (dx <= abs(tol_x)) & (dy <= abs(tol_y)) & (dz <= abs(tol_z))

    for data_var in data_vars if isinstance(data_vars, list) else [data_vars]:
        collection_data[data_var] = np.where(mask, result[data_var], np.nan)

    return Collection(
        collection.data.assign(**collection_data[data_vars]),
        header=collection.header.copy(),
        has_inclined=collection.has_inclined,
        vertical_datum=collection.vertical_datum,
    )


def add_model_data(
    survey_data: Collection | pd.DataFrame,
    model: xr.Dataset | xr.DataArray,
    *,
    data_vars: str | list[str] = None,
    suffix: str = None,
    agg_funcs: dict[str, str] = None,
    nr_: str = "nr",
    surface_: str = "surface",
    bottom_: str = "bottom",
) -> Collection | pd.DataFrame:
    """
    Add information from a variable of a model as a column to the data
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
    model : xr.Dataset | xr.DataArray
        Model data to add the information from to the `Collection` instance.
    variable : str
        Name of the variable in the model to add.

    Returns
    -------
    :class:`~geost.base.Collection`
        `Collection` instance with the information from the model variable added
        to the data table.

    """
    from geost.models._core import ModelType

    if return_collection := isinstance(survey_data, Collection):
        header = survey_data.header
        data = survey_data.data
    elif isinstance(survey_data, pd.DataFrame):
        header = survey_data.gst.to_header()
        data = survey_data
    else:
        raise TypeError("survey_data must be either a Collection or a pd.DataFrame")

    if data_vars is None:
        variable = (
            list(model.data_vars) if isinstance(model, xr.Dataset) else [model.name]
        )
    else:
        variable = [data_vars] if isinstance(data_vars, str) else data_vars

    agg_funcs = agg_funcs or {}
    if suffix is not None:
        result_vars = [f"{var}{suffix}" for var in variable]
        extra = [f"{key}{suffix}" for key in agg_funcs.keys()]
    else:
        result_vars = variable
        extra = list(agg_funcs.keys())
    result_vars += extra

    data.drop(
        columns=result_vars, inplace=True, errors="ignore"
    )  # Drop existing column if present to avoid conflicts

    if model.gst.model_type == ModelType.VOXEL:
        var_df = _get_voxelmodel_df(
            model, header, nr_, bottom_, surface_, variable, agg_funcs
        )
    else:
        raise NotImplementedError("Only voxel models are currently supported.")

    variable = variable + list(agg_funcs.keys())
    var_df.rename(
        columns={v: rv for v, rv in zip(variable, result_vars, strict=True)},
        inplace=True,
    )
    result = data.gst.merge_sorted(var_df[[nr_, bottom_, *result_vars]], backfill=True)
    result.dropna(
        subset=surface_, inplace=True
    )  # Rows with NaN in the surface column are rows where the model is deeper than the survey

    if return_collection:
        result = Collection(
            result,
            header=header.copy(),
            has_inclined=survey_data.has_inclined,
            vertical_datum=survey_data.vertical_datum,
        )
    return result


def _get_voxelmodel_df(model, header, nr_, bottom_, surface_, variable, agg_funcs):
    *_, dz = model.gst.resolution()

    var_select = model.gst.select_points(header)
    var_select = var_select.rename({model.gst.z_dim: bottom_})
    var_select = var_select.assign_coords(
        {bottom_: var_select[bottom_] - (0.5 * dz)}
    )  # Translate voxel mid depths to bottom depths
    var_select[nr_] = (("idx"), header[nr_].loc[var_select["idx"]])
    var_select[surface_] = (("idx"), header[surface_].loc[var_select["idx"]])

    var_df = _create_dataframe_and_reduce(
        var_select,
        nr=nr_,
        bottom=bottom_,
        surface=surface_,
        variable=variable,
        agg_funcs=agg_funcs,
    )
    return var_df


def _get_layermodel_df():
    pass


def _create_dataframe_and_reduce(
    ds: xr.Dataset | xr.DataArray,
    nr: str,
    bottom: str,
    surface: str,
    variable: str | list,
    agg_funcs: dict[str, str],
) -> pd.DataFrame:
    """
    Helper for `add_voxelmodel_variable` to reduce the selection DataArray from
    `model.gst.select_points` to a DataFrame containing relevant layer boundaries.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Selection result of `model.gst.select_points`.
    nr : str
        Name of the survey ID column.
    bottom : str
        Name of the bottom depth column.
    surface : str
        Name of the surface depth column.
    variable : str | list
        Name or names of the variable(s) to add from the voxelmodel, these are treated
        as consecutive layers to add.

    Returns
    -------
    pd.DataFrame

    """
    var_df = ds.to_dataframe().reset_index()
    var_df = var_df.dropna(subset=variable).sort_values(
        by=[nr, bottom], ascending=[True, False]
    )
    var_df = var_df.gst.aggregate_consecutive_layers(variable, agg_funcs)
    var_df = var_df[
        var_df[bottom] < var_df[surface]
    ]  # Only keep layers below surface, strat boundaries are bottoms of layers
    var_df[bottom] = var_df[surface] - var_df[bottom]

    return var_df
