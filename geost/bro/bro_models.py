from pathlib import WindowsPath
from typing import List

import rioxarray as rio
import xarray as xr

from geost.models import LayerModel, VoxelModel

from .bro_utils import coordinates_to_cellcenters, flip_ycoordinates


class GeoTop(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | WindowsPath,
        data_vars: List[str] = None,
        bbox: tuple[float, float, float, float] = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read the BRO GeoTop subsurface model from a netcdf dataset into a GeoTop
        VoxelModel instance. The complete dataset of GeoTop can be downloaded from:
        https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : str | WindowsPath
            Path to the netcdf file of GeoTop.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area of
            GeoTop. The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        **xr_kwargs
            Additional keyword arguments xarray.open_dataset. See relevant documentation
            for details.

        Returns
        -------
        GeoTop
            GeoTop instance of the netcdf file.

        """
        cellsize = 100
        dz = 0.5
        crs = 28992

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print("Load data")
            ds = ds.load()

        ds = flip_ycoordinates(ds)
        ds = ds.rio.write_crs(crs)

        return cls(ds)

    @classmethod
    def from_opendap(
        cls,
        url: str = r"https://dinodata.nl/opendap/GeoTOP/geotop.nc",
        data_vars: List[str] = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Download an area of GeoTop directly from the OPeNDAP data server into a GeoTop
        VoxelModel instance.

        Parameters
        ----------
        url : str
            Url to the netcdf file on the OPeNDAP server. See:
            https://www.dinoloket.nl/modelbestanden-aanvragen
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area of
            GeoTop. The default is None but for practical reasons, specifying a bounding
            box is advised (TODO: find max downloadsize for server).
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        **xr_kwargs
            Additional keyword arguments xarray.open_dataset. See relevant documentation
            for details.

        Returns
        -------
        GeoTop
            GeoTop instance for the selected area.

        """
        return cls.from_netcdf(url, data_vars, bbox, lazy, **xr_kwargs)


class Regis(LayerModel):
    pass
