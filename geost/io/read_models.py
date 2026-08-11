from pathlib import Path

import xarray as xr


def read_model_netcdf(
    nc_file: str | Path,
    data_vars: str | list[str] = None,
    bbox: tuple[float, float, float, float] = None,
    load: bool = False,
    **xr_kwargs,
) -> xr.Dataset | xr.DataArray:
    """
    Read data from a NetCDF file of a voxelmodel or layermodel data into an xarray Dataset
    or DataArray.

    Parameters
    ----------
    nc_file : str | Path
        Path to the netcdf file of the voxelmodel or layermodel.
    data_vars : str | list[str], optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple (xmin, ymin, xmax, ymax), optional
        Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area. The
        default is None. If bbox is None and the dataset is large, it is recommended to
        use lazy loading (load=False) and specify chunks (see examples below) to avoid
        memory issues.
    load : bool, optional
        If True, the netcdf file is loaded into memory immediately. This will improve the
        speed of several analyses but will cause higher memory usage or memory error if
        the dataset is too large. Use False for lazy loading, which allows to process data
        that does not fit into memory. The default is False.
    **xr_kwargs
        Additional keyword arguments xarray.open_dataset. See relevant documentation
        for details.

    Returns
    -------
    xr.Dataset | xr.DataArray
        xarray Dataset instance of the netcdf file or DataArray if a single variable is
        selected.

    Examples
    --------
    Read all model data from a local NetCDF file:

    >>> import geost
    >>> model = geost.read_model_netcdf("my_netcdf_file.nc")

    Read one or more data variables within a specific area from the NetCDF file and directly
    load the data into memory:

    >>> model = geost.read_model_netcdf(
    ...     "my_netcdf_file.nc", data_vars="my_var", bbox=(1, 1, 3, 3), load=True
    ... )
    >>> model = geost.read_model_netcdf(
    ...     "my_netcdf_file.nc",
    ...     data_vars=["my_var", "my_other_var"],
    ...     bbox=(1, 1, 3, 3),
    ...     load=True
    ... )

    Read the entire model data but specify chunks to avoid memory issues when the dataset
    is large:

    >>> model = geost.read_model_netcdf(
    ...     "my_netcdf_file.nc",
    ...     chunks={"x": 100, "y": 100, "z": -1} # -1 takes the entire "z" dimension in a chunk
    ... )

    """
    ds = xr.open_dataset(nc_file, **xr_kwargs)

    if bbox:
        ds = ds.gst.slice_xy(*bbox)

    if data_vars:
        ds = ds[data_vars]

    if load:
        ds.load()

    return ds


def read_model_from_opendap(  # pragma: no cover
    url: str,
    data_vars: list[str] = None,
    bbox: tuple[float, float, float, float] = None,
    load: bool = False,
    **xr_kwargs,
) -> xr.Dataset | xr.DataArray:
    """
    Read NetCDF data from an OpenDAP server of a voxelmodel or layermodel data into an
    xarray Dataset or DataArray. Note that this function is a wrapper around the
    :func:`geost.read_model_netcdf` function and uses it to read the data.

    Parameters
    ----------
    url : str
        URL to the OpenDAP NetCDF file of the voxelmodel or layermodel.
    data_vars : str | list[str], optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple (xmin, ymin, xmax, ymax), optional
        Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area. The
        default is None. If bbox is None and the dataset is large, it is recommended to
        use lazy loading (load=False) and specify chunks (see examples below) to avoid
        memory issues.
    load : bool, optional
        If True, the netcdf file is loaded into memory immediately. This will improve the
        speed of several analyses but will cause higher memory usage or memory error if
        the dataset is too large. Use False for lazy loading, which allows to process data
        that does not fit into memory. The default is False.
    **xr_kwargs
        Additional keyword arguments xarray.open_dataset. See relevant documentation
        for details.

    Returns
    -------
    xr.Dataset | xr.DataArray
        xarray Dataset instance of the netcdf file or DataArray if a single variable is
        selected.

    Examples
    --------
    Read all model data from an OpenDAP server in chunks:

    >>> import geost
    >>> model = geost.read_model_from_opendap(
    ...     "https://opendap.example.org/data/model.nc"
    ...     chunks={"x": 100, "y": 100, "z": -1} # -1 takes the entire "z" dimension in a chunk
    ... )

    """
    return read_model_netcdf(
        url, data_vars=data_vars, bbox=bbox, load=load, **xr_kwargs
    )
