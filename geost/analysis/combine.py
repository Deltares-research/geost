from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xarray as xr

from geost.base import Collection


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
    aggregate_vars: dict[str, str] = None,
    suffix: str = None,
    nr_: str = "nr",
    surface_: str = "surface",
    bottom_: str = "bottom",
) -> Collection | pd.DataFrame:
    """
    Add information from one or more model variables as columns to survey data.

    For each survey, this determines the vertical model stack at the survey's location
    and identifies the layer boundaries within that stack based on changes in the selected
    model variables. Survey intervals are split at these model layer boundaries, and the
    corresponding model information is backfilled to the resulting intervals. The resulting
    data are sorted by depth. This is illustrated in the example below.

    .. code-block:: text

        Survey data:
           nr  top  bottom lith
        0   A  0.0    10.0    Z

        Layer boundaries derived from the model data for "variable" at the location of the
        survey:
           top  bottom  variable
        0  0.0     8.0         1
        1  8.0    11.0         2

        Result:
           nr  top  bottom lith  variable
        0   A  0.0     8.0    Z       1.0
        1   A  8.0    10.0    Z       2.0

    Parameters
    ----------
    survey_data : :class:`~geost.base.Collection` | pd.DataFrame
        `Collection` or `DataFrame` to add the model data to.
    model : xr.Dataset | xr.DataArray
        Xarray Dataset or DataArray containing the model data.
    data_vars : str | list[str], optional
        Variable or variables from the model to add. These variables are used to determine
        the layer boundaries. A layer boundary is defined where the value of a variable
        changes with depth; consecutive equal values are considered part of the same layer.
        If multiple variables are given, they are treated jointly when determining the
        layer boundaries. If None, all model variables are added and jointly considered
        for determining the layer boundaries.
    aggregate_vars : dict[str, str], optional
        Optional dictionary specifying how to aggregate additional model variables, not
        given in `data_vars`, over the layers derived from `data_vars`. The keys are the
        model variable names and the values are the aggregation functions (e.g., `"mean"`
        or `"sum"`).
    suffix : str, optional
        Suffix to append to the added model variable columns to avoid name conflicts with
        existing columns in the survey data. If None, no suffix is added. In case of a name
        conflict, the existing column is overwritten.

    Returns
    -------
    :class:`~geost.base.Collection` | pd.DataFrame
        `Collection` or `pandas.DataFrame` instance with the information from the model
        added to the data.

    """
    aggregate_vars = aggregate_vars or {}

    if return_collection := isinstance(survey_data, Collection):
        header = survey_data.header.copy()
        data = survey_data.data.copy()
    else:
        header = survey_data.gst.to_header()
        data = survey_data.copy()

    if data_vars is None:
        variable = (
            list(model.data_vars) if isinstance(model, xr.Dataset) else [model.name]
        )
    else:
        variable = [data_vars] if isinstance(data_vars, str) else data_vars

    mdf = _get_model_dataframe(
        model, header, variable, aggregate_vars, nr_, bottom_, surface_
    )

    if suffix is not None:
        all_vars = list(itertools.chain(variable, aggregate_vars.keys()))
        result_vars = [f"{var}{suffix}" for var in all_vars]
        mdf.rename(columns=dict(zip(all_vars, result_vars)), inplace=True)
    else:
        result_vars = variable.copy() + list(aggregate_vars.keys())

    data.drop(
        columns=result_vars, inplace=True, errors="ignore"
    )  # Drop existing columns to avoid conflicts with new model data

    temp_nodata = -999999
    result = data.gst.merge_sorted(
        mdf[[nr_, surface_, bottom_, *result_vars]].fillna(temp_nodata), # Use `temp_nodata` for correct backfill
        backfill=True,
        drop_overlapping_columns=True,
    )  # fmt: skip
    result.dropna(
        subset=result.gst._x, inplace=True
    )  # Rows with NaN in a coordinate column are rows where the model is deeper than the survey
    result[result_vars] = result[result_vars].where(result[result_vars] != temp_nodata)

    if return_collection:
        result = Collection(
            result,
            header=header,
            has_inclined=survey_data.has_inclined,
            vertical_datum=survey_data.vertical_datum,
        )

    return result


def _get_model_dataframe(
    model: xr.Dataset | xr.DataArray,
    header: pd.DataFrame,
    variable: str | list,
    agg_funcs: dict[str, str],
    nr_: str,
    bottom_: str,
    surface_: str,
) -> pd.DataFrame:
    """
    Helper function for `add_model_data` the select a model at locations of survey data
    and reduce the result to a pandas.DataFrame that can be merged with the survey data.

    """
    from geost.models._core import ModelType

    sel = model.gst.select_points(header)
    if model.gst.model_type == ModelType.VOXEL:
        sel = sel.rename({model.gst.z_dim: bottom_})

        *_, dz = model.gst.resolution()
        sel = sel.assign_coords(
            {bottom_: sel[bottom_] - (0.5 * dz)}
        )  # Translate voxel mid depths to bottom depths
    else:
        sel = sel.rename({model.gst._bottom: bottom_})

    sel[nr_] = (("idx"), header[nr_].loc[sel["idx"]])
    sel[surface_] = (("idx"), header[surface_].loc[sel["idx"]])

    return _create_dataframe_and_reduce(
        sel,
        variable=variable,
        agg_funcs=agg_funcs,
        nr=nr_,
        bottom=bottom_,
        surface=surface_,
    )


def _create_dataframe_and_reduce(
    ds: xr.Dataset | xr.DataArray,
    variable: str | list,
    agg_funcs: dict[str, str],
    nr: str,
    bottom: str,
    surface: str,
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
