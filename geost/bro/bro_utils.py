import numpy as np
import xarray as xr


def get_bbox_criteria(xmin, ymin, xmax, ymax):
    json_line = {
        "area": {
            "boundingBox": {
                "lowerCorner": {"lat": ymin, "lon": xmin},
                "upperCorner": {"lat": ymax, "lon": xmax},
            }
        }
    }
    return json_line


def coordinates_to_cellcenters(
    ds: xr.Dataset | xr.DataArray, cellsize: int | float, dz: int | float = None
) -> xr.Dataset | xr.DataArray:
    """
    Change the coordinates of an Xarray Dataset or DataArray to cellcenters. This assumes
    that each coordinate in the Dataset are lower left, lower z (if "z" dimension is present)
    coordinates and y-coordinates are in increasing order which is the case in BRO distributed
    models.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Dataset or DataArray to change the coordinates for.
    cellsize : int | float
        Cellsize of the Dataset or DataArray.
    dz : int | float, optional
        Size of the "z" dimension if present in the Dataset or DataArray. If a "z" dimension
        is present but dz is None, the dimension is ignored. The default is None.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Dataset or DataArray with the changed coordinates.

    """
    ds["x"] = ds["x"] + (cellsize / 2)
    ds["y"] = ds["y"] + (cellsize / 2)
    if dz:
        ds["z"] = ds["z"] + (dz / 2)
    return ds


def flip_ycoordinates(ds):
    if "z" in ds.dims:
        ds = ds.transpose("y", "x", "z")
    else:
        ds = ds.transpose("y", "x")

    if ds["y"][-1] > ds["y"][0]:
        ds = ds.sel(y=slice(None, None, -1))

    return ds
