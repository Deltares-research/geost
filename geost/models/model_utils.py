import numpy as np
import xarray as xr
from shapely.geometry import LineString


def sample_with_coords(ds: xr.Dataset | xr.DataArray, coords: np.ndarray):
    """
    Sample x and y dims of an Xarray Dataset or DataArray using an array of x and y
    coordinates. Only coordinates that are within the extent of the Dataset or DataArray
    are selected.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to sample. Must contain dimensions 'x' and 'y' that
        refer to the coordinates.
    coords : np.ndarray
        Numpy array with shape (n, 2) of the x and y coordinates to select.

    Returns
    -------
    ds_sel : xr.Dataset or xr.DataArray
        Sampled Dataset or DataArray with dimension 'idx' which is the index of the input
        coordinates which are selected.

    """
    x, y = coords[:, 0], coords[:, 1]
    idx = np.arange(len(coords))

    xmin, ymin, xmax, ymax = ds.rio.bounds()
    outside_bbox = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)

    idx = idx[~outside_bbox]
    x = x[~outside_bbox]
    y = y[~outside_bbox]

    ds_sel = ds.sel(
        x=xr.DataArray(x, dims="idx"), y=xr.DataArray(y, dims="idx"), method="nearest"
    )
    ds_sel = ds_sel.assign_coords(idx=("idx", idx))
    return ds_sel


def sample_along_line(
    ds: xr.Dataset | xr.DataArray,
    line: LineString,
    dist: int | float = None,
    nsamples: int = None,
):
    """
    Sample x and y dims of an Xarray Dataset or DataArray over distance along a
    Shapely LineString object. Sampling can be done using a specified distance
    or a specified number of samples that need to be taken along the line.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to sample. Must contain dimensions 'x' and 'y' that
        refer to the coordinates.
    line : LineString
        shapely.geometry.LineString object to use for sampling.
    dist : int, float, optional
        Distance between each sample along the line. Takes equally distant samples
        from the start of the line untill it reaches the end. The default is None.
    nsamples : int, optional
        Number of samples to take along the line between the beginning and the end.
        The default is None.

    Raises
    ------
    ValueError
        If both or none of dist and nsamples are specified.

    Returns
    -------
    ds_sel : xr.Dataset or xr.DataArray
        Sampled Dataset or DataArray with dimension 'dist' for distance.

    """
    if dist and nsamples:
        raise ValueError("Cannot use 'dist' and 'nsamples' together, use one option.")

    elif dist:
        sample_locs = np.arange(0, line.length, dist)

    elif nsamples:
        sample_locs = np.linspace(0, line.length, nsamples)

    else:
        raise ValueError("'dist' or 'nsamples' not specified, use one option.")

    samplepoints = np.array([_interpolate_point(line, loc) for loc in sample_locs])
    dist, x, y = samplepoints[:, 0], samplepoints[:, 1], samplepoints[:, 2]

    ds_sel = ds.sel(
        x=xr.DataArray(x, dims="dist"), y=xr.DataArray(y, dims="dist"), method="nearest"
    )
    ds_sel = ds_sel.assign_coords(dist=("dist", dist))

    return ds_sel


def _interpolate_point(line, loc):
    """
    Return the location (i.e. distance) and x and y coordinates of an interpolated
    point along a Shapely LineString object.

    Parameters
    ----------
    line : LineString
        shapely.geometry.LineString object.
    loc : int, float
        Distance along the LineString to interpolate the point at.

    """
    p = line.interpolate(loc)
    return loc, p.x, p.y


def label_consecutive_2d(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Label consecutive array elements with unique numbers along a specified axis in a 2D
    array.

    Parameters
    ----------
    array : np.ndarray
        Array of shape (m, n) to label the elements in.
    axis : int, optional
        Axis along to label. The default is 0.

    Returns
    -------
    np.ndarray
        Array with labelled elements with the same shape as the input array.

    """
    labels = np.cumsum(np.diff(array, axis=axis) != 0, axis=axis)

    if axis == 0:
        start_labels = np.zeros(array.shape[1], dtype=array.dtype).reshape(1, -1)
        labels = np.r_[start_labels, labels]
    else:
        start_labels = np.zeros(array.shape[0], dtype=array.dtype).reshape(-1, 1)
        labels = np.c_[start_labels, labels]

    return labels
