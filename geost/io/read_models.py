from pathlib import Path

import xarray as xr


def _prepare_dataset(
    ds: xr.Dataset,
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    load: bool = False,
) -> xr.Dataset | xr.DataArray:
    """
    Helper for reader functions to prepare the dataset by selecting data variables, slicing
    the dataset based on the provided bounding box and optionally loading it into memory.

    """
    if bbox is not None:
        ds = ds.gst.slice_xy(*bbox)

    if data_vars is not None:
        ds = ds[data_vars]

    if load:
        ds.load()

    return ds


def read_model_netcdf(
    nc_file: str | Path,
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
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
    data_vars : str | list[str] | None, optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple[float, float, float, float] | None, optional
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
    return _prepare_dataset(ds, data_vars=data_vars, bbox=bbox, load=load)


def read_model_from_opendap(  # pragma: no cover
    url: str,
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
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
    data_vars : str | list[str] | None, optional
        List of data variable names or a single data variable name specifying which data
        variables to return. If None, all data variables are returned.
    bbox : tuple[float, float, float, float] | None, optional
        Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area. The
        default is None. If bbox is None and the dataset is large, it is recommended to
        use lazy loading (load=False) and specify chunks (see examples below) to avoid
        memory issues.
    load : bool, optional
        If True, the netcdf file is loaded into memory immediately. This will improve the
        speed of several analyses but will cause higher memory usage or memory error if
        the dataset is too large. Use False for lazy loading, which allows to process data
        that does not fit into memory. The default is False. Note that if `load=True` is
        used and `chunks` are specified in `xr_kwargs`, the chunks will be ignored and the
        entire dataset will be loaded into memory as Numpy Arrays.
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
    ...     "https://opendap.example.org/data/model.nc",
    ...     chunks={"x": 100, "y": 100, "z": -1}, # -1 takes the entire "z" dimension in a chunk
    ... )

    """
    return read_model_netcdf(
        url, data_vars=data_vars, bbox=bbox, load=load, **xr_kwargs
    )


def read_geotop_netcdf(
    nc_file: str | Path,
    *,
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    load: bool = False,
    **xr_kwargs,
) -> xr.Dataset:
    """
    Read GeoTOP NetCDF data into GeoST compatible xarray Dataset or DataArray.

    Parameters
    ----------
    nc_file : str | Path
        Path to the GeoTOP netcdf file.
    data_vars : str | list[str] | None, optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple[float, float, float, float] | None, optional
        Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area. The
        default is None. If bbox is None and the dataset is large, it is recommended to
        use lazy loading (load=False) and specify chunks (see examples below) to avoid
        memory issues.
    load : bool, optional
        If True, the netcdf file is loaded into memory immediately. This will improve the
        speed of several analyses but will cause higher memory usage or memory error if
        the dataset is too large. Use False for lazy loading, which allows to process data
        that does not fit into memory. The default is False.
    **xr_kwargs : Any
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
    >>> model = geost.read_geotop_netcdf("my_netcdf_file.nc")

    Read one or more data variables within a specific area from the NetCDF file and directly
    load the data into memory:

    >>> model = geost.read_geotop_netcdf(
    ...     "my_netcdf_file.nc", data_vars="my_var", bbox=(1, 1, 3, 3), load=True
    ... )
    >>> model = geost.read_geotop_netcdf(
    ...     "my_netcdf_file.nc",
    ...     data_vars=["my_var", "my_other_var"],
    ...     bbox=(1, 1, 3, 3),
    ...     load=True
    ... )

    Read the entire model data but specify chunks to avoid memory issues when the dataset
    is large:

    >>> model = geost.read_geotop_netcdf(
    ...     "my_netcdf_file.nc",
    ...     chunks={"x": 100, "y": 100, "z": -1} # -1 takes the entire "z" dimension in a chunk
    ... )

    """

    def _shift_coordinates(ds):
        """
        GeoTOP coordinates are lowerleft bottom of each voxel, we shift towards the
        centre coordinates.

        """
        xres, yres, zres = ds.gst.resolution()
        x_dim, y_dim, z_dim = ds.gst.x_dim, ds.gst.y_dim, ds.gst.z_dim
        return ds.assign_coords(
            {
                x_dim: ds[x_dim] + (xres / 2),
                y_dim: ds[y_dim] + (yres / 2),
                z_dim: ds[z_dim] + (zres / 2),
            }
        )

    ds = xr.open_dataset(nc_file, **xr_kwargs)
    ds.gst.write_crs(28992, inplace=True)
    ds = _shift_coordinates(ds)
    return _prepare_dataset(ds, data_vars=data_vars, bbox=bbox, load=load)


def read_geotop_from_opendap(  # pragma: no cover
    *,
    url: str = r"https://www.dinodata.nl/opendap/GeoTOP/geotop.nc",
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    load: bool = False,
    **xr_kwargs,
) -> xr.Dataset:
    """
    Read GeoTOP NetCDF data from an OpenDAP server into GeoST compatible xarray Dataset
    or DataArray.

    Parameters
    ----------
    url : str
        URL to the GeoTOP netcdf file on the OPeNDAP server. See:
        https://www.dinoloket.nl/modelbestanden-aanvragen. The default is
        "https://www.dinodata.nl/opendap/GeoTOP/geotop.nc".
    data_vars : str | list[str] | None, optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple[float, float, float, float] | None, optional
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
    Read "strat" and "lithok" variables within a specific bounding box from the TNO
    OpenDAP server:

    >>> import geost
    >>> geotop = geost.read_geotop_from_opendap(
    ...     bbox=(110_000, 440_000, 120_000, 450_000),
    ...     data_vars=["lithok", "strat"],
    ...     chunks={"x": 100, "y": 100, "z": -1} # -1 takes the entire "z" dimension in a chunk
    ... )

    """
    return read_geotop_netcdf(
        url, data_vars=data_vars, bbox=bbox, load=load, **xr_kwargs
    )


def read_regis_netcdf(
    nc_file: str | Path,
    *,
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    load: bool = False,
    **xr_kwargs,
) -> xr.Dataset:
    """
    Read REGIS NetCDF data into GeoST compatible xarray Dataset or DataArray.

    Parameters
    ----------
    nc_file : str | Path
        Path to the REGIS netcdf file.
    data_vars : str | list[str] | None, optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple[float, float, float, float] | None, optional
        Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area. The
        default is None. If bbox is None and the dataset is large, it is recommended to
        use lazy loading (load=False) and specify chunks (see examples below) to avoid
        memory issues.
    load : bool, optional
        If True, the netcdf file is loaded into memory immediately. This will improve the
        speed of several analyses but will cause higher memory usage or memory error if
        the dataset is too large. Use False for lazy loading, which allows to process data
        that does not fit into memory. The default is False.
    **xr_kwargs : Any
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
    >>> model = geost.read_regis_netcdf("my_netcdf_file.nc")

    Read one or more data variables within a specific area from the NetCDF file and directly
    load the data into memory:

    >>> model = geost.read_regis_netcdf(
    ...     "my_netcdf_file.nc", data_vars="my_var", bbox=(1, 1, 3, 3), load=True
    ... )
    >>> model = geost.read_regis_netcdf(
    ...     "my_netcdf_file.nc",
    ...     data_vars=["my_var", "my_other_var"],
    ...     bbox=(1, 1, 3, 3),
    ...     load=True
    ... )

    Read the entire model data but specify chunks to avoid memory issues when the dataset
    is large:

    >>> model = geost.read_regis_netcdf(
    ...     "my_netcdf_file.nc",
    ...     chunks={"x": 100, "y": 100, "layer": -1} # -1 takes the entire "layer" dimension in a chunk
    ... )

    """
    ds = xr.open_dataset(nc_file, **xr_kwargs)
    ds.gst.write_crs(28992, inplace=True)

    x_dim, y_dim, layer = ds.gst.x_dim, ds.gst.y_dim, ds.gst.z_dim
    ds = ds.assign_coords(
        {
            x_dim: ds["x_bounds"].sum(axis=1) / 2,
            y_dim: ds["y_bounds"].sum(axis=1) / 2,
            layer: ds[layer].str.decode("utf-8"),
        }
    ).drop_vars(
        ["x_bounds", "y_bounds", "lat_bounds", "lon_bounds", "lat", "lon"]
    )  # These variables are not needed for further analysis and can cause errors

    if data_vars is not None:
        # If data_vars is specified, ensure that the bottom and top variables are included
        if ds.gst._bottom not in data_vars:
            data_vars = [ds.gst._bottom] + data_vars
        if ds.gst._top not in data_vars:
            data_vars = [ds.gst._top] + data_vars

    ds = _prepare_dataset(ds, data_vars=data_vars, bbox=bbox, load=load)
    ds = ds.sel({layer: ds[layer] != "mv"})

    return ds


def read_regis_from_opendap(  # pragma: no cover
    *,
    url: str = r"https://www.dinodata.nl/opendap/REGIS/REGIS.nc",
    data_vars: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    load: bool = False,
    **xr_kwargs,
) -> xr.Dataset | xr.DataArray:
    """
    Read REGIS NetCDF data into GeoST compatible xarray Dataset or DataArray.

    Parameters
    ----------
    url : str
        URL to the REGIS netcdf file.
    data_vars : str | list[str] | None, optional
        List of data variable names or a single data variable name specifying which data
        variables to return.
    bbox : tuple[float, float, float, float] | None, optional
        Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area. The
        default is None. If bbox is None and the dataset is large, it is recommended to
        use lazy loading (load=False) and specify chunks (see examples below) to avoid
        memory issues.
    load : bool, optional
        If True, the netcdf file is loaded into memory immediately. This will improve the
        speed of several analyses but will cause higher memory usage or memory error if
        the dataset is too large. Use False for lazy loading, which allows to process data
        that does not fit into memory. The default is False.
    **xr_kwargs : Any
        Additional keyword arguments xarray.open_dataset. See relevant documentation
        for details.

    Returns
    -------
    xr.Dataset | xr.DataArray
        xarray Dataset instance of the netcdf file or DataArray if a single variable is
        selected.

    Examples
    --------
    Read "kD" variable within a specific bounding box from the from the TNO OpenDAP server:

    >>> import geost
    >>> regis = geost.read_regis_from_opendap(
    ...     bbox=(110_000, 440_000, 120_000, 450_000),
    ...     data_vars="kD",
    ...     chunks={"x": 100, "y": 100, "layer": -1} # -1 takes the entire "layer" dimension in a chunk
    ... )

    """
    return read_regis_netcdf(
        url, data_vars=data_vars, bbox=bbox, load=load, **xr_kwargs
    )
