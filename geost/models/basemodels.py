import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr

from geost.export import vtk
from geost.utils import check_geometry_instance

from .model_utils import sample_along_line, sample_with_coords


class AbstractSpatial(ABC):  # pragma: no cover
    @property
    @abstractmethod
    def xmin(self):
        pass

    @property
    @abstractmethod
    def xmax(self):
        pass

    @property
    @abstractmethod
    def ymin(self):
        pass

    @property
    @abstractmethod
    def ymax(self):
        pass

    @property
    @abstractmethod
    def resolution(self):
        pass

    @property
    @abstractmethod
    def crs(self):
        pass

    @abstractmethod
    def sel(self):
        pass

    @abstractmethod
    def isel(self):
        pass

    @abstractmethod
    def select_with_line(self):
        pass

    @abstractmethod
    def select_with_points(self):
        pass

    @abstractmethod
    def select_within_polygons(self):
        pass

    @abstractmethod
    def select_within_bbox(self):
        pass

    @abstractmethod
    def select_by_values(self):
        pass


class AbstractModel3D(ABC):  # pragma: no cover
    @property
    @abstractmethod
    def zmin(self):
        pass

    @property
    @abstractmethod
    def zmax(self):
        pass

    @abstractmethod
    def select_top(self):
        pass

    @abstractmethod
    def select_bottom(self):
        pass

    @abstractmethod
    def slice_depth_interval(self):
        pass

    @abstractmethod
    def select_surface_level(self):
        pass

    @abstractmethod
    def zslice_to_tiff(self):
        pass

    @abstractmethod
    def get_thickness(self):
        pass

    @abstractmethod
    def to_pyvista_grid(self):
        pass


class VoxelModel(AbstractSpatial, AbstractModel3D):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def __getitem__(self, item):
        return self.ds[item]

    def __setitem__(self, key, item):
        self.ds[key] = item

    def __repr__(self):
        instance = f"{self.__class__.__name__}"
        layers = self.ds.data_vars
        dimensions = f"Dimensions: {dict(self.ds.sizes)}"
        resolution = f"Resolution (y, x, z): {self.resolution}"
        return f"{instance}\n{layers}\n{dimensions}\n{resolution}"

    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | Path,
        data_vars: list[str] = None,
        bbox: tuple[float, float, float, float] = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read data from a NetCDF file of a voxelmodel data into a VoxelModel instance.

        This assumes the voxelmodel is according to the following conventions:
        - The coordinate dimensions of the voxelmodel are: "x", "y" (horizontal) and "z"
        (depth).
        - The coordinates in the y-dimension are in descending order.

        Parameters
        ----------
        nc_path : str | Path
            Path to the netcdf file of the voxelmodel.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area of
            the voxelmodel. The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        **xr_kwargs
            Additional keyword arguments xarray.open_dataset. See relevant documentation
            for details.

        Returns
        -------
        VoxelModel
            VoxelModel instance of the netcdf file.

        Examples
        --------
        Read all model data from a local NetCDF file:

        >>> VoxelModel.from_netcdf("my_netcdf_file.nc")

        Read specific data variables and the data within a specific area from the NetCDF
        file:

        >>> VoxelModel.from_netcdf(
        ...     "my_netcdf_file.nc",
        ...     data_vars=["my_var"],
        ...     bbox=(1, 1, 3, 3) # (xmin, ymin, xmax, ymax)
        ... )

        Note that this method assumes the y-coordinates are in descending order. For y-
        ascending coordinates change ymin and ymax coordinates:

        >>> VoxelModel.from_netcdf(
        ...     "my_netcdf_file.nc", bbox=(1, 3, 1, 3) # (xmin, ymax, xmax, ymin)
        ... )

        """
        ds = xr.open_dataset(nc_path, **xr_kwargs)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print("Load data")
            ds = ds.load()

        return cls(ds)

    @property
    def xmin(self) -> float:
        return self.horizontal_bounds[0]

    @property
    def xmax(self) -> float:
        return self.horizontal_bounds[2]

    @property
    def ymin(self) -> float:
        return self.horizontal_bounds[1]

    @property
    def ymax(self) -> float:
        return self.horizontal_bounds[3]

    @property
    def zmin(self) -> float:
        return self.vertical_bounds[0]

    @property
    def zmax(self) -> float:
        return self.vertical_bounds[1]

    @property
    def horizontal_bounds(self) -> tuple[float, float, float, float]:
        return self.ds.rio.bounds()

    @property
    def vertical_bounds(self) -> tuple[float, float]:
        if not hasattr(self, "_zmin"):
            self._get_internal_zbounds()
        return float(self._zmin), float(self._zmax)

    @property
    def resolution(self) -> tuple[float, float, float]:
        """
        Return a tuple (dy, dx, dz) of the VoxelModel resolution.
        """
        if not hasattr(self, "_dz"):
            self._get_internal_zbounds()
        dx, dy = np.abs(self.ds.rio.resolution())
        return (float(dy), float(dx), float(self._dz))

    @property
    def sizes(self):
        return self.ds.sizes

    @property
    def shape(self):
        return tuple(self.sizes.values())

    @property
    def crs(self):
        return self.ds.rio.crs

    @property
    def variables(self):
        return self.ds.data_vars

    def _get_internal_zbounds(self):
        self._dz = np.abs(np.diff(self["z"])[0])
        self._zmin = np.min(self["z"].values)
        self._zmax = np.max(self["z"].values)
        self._zmin -= 0.5 * self._dz
        self._zmax += 0.5 * self._dz

    def sel(self, **xr_kwargs):
        """
        Use Xarray selection functionality to select indices along specified dimensions.
        This uses the ".sel" method of an Xarray Dataset.

        Parameters
        ----------
        **xr_kwargs
            xr.Dataset.sel keyword arguments. See relevant Xarray documentation.

        Examples
        --------
        Select a specified coordinates or slice coordinates from the VoxelModel instance:

        >>> selected = voxelmodel.sel(x=[1, 2, 3])  # Using keyword arguments
        >>> selected = voxelmodel.sel({"x": [1, 2, 3]})  # Using a dictionary
        >>> selected = voxelmodel.sel(x=slice(1, 4))  # Using a slice

        Using additional options as well. For instance, when the desired coordinates do
        not exactly match the VoxelModel coordinates, select the nearest:

        >>> selected = voxelmodel.sel(x=[1.1, 2.5, 3.3], method="nearest")

        """
        selected = self.ds.sel(**xr_kwargs)
        return self.__class__(selected)

    def isel(self, **xr_kwargs):
        """
        Use Xarray selection functionality to select indices along specified dimensions.
        This uses the ".isel" method of an Xarray Dataset.

        Parameters
        ----------
        **xr_kwargs
            xr.Dataset.isel keyword arguments. See relevant Xarray documentation.

        Examples
        --------
        Select a specified coordinates or slice coordinates from the VoxelModel instance:

        >>> selected = voxelmodel.isel(x=[1, 2, 3])  # Using keyword arguments
        >>> selected = voxelmodel.isel({"x": [1, 2, 3]})  # Using a dictionary
        >>> selected = voxelmodel.isel(x=slice(1, 4))  # Using a slice

        """
        selected = self.ds.isel(**xr_kwargs)
        return self.__class__(selected)

    def select_with_points(self, points: str | Path | gpd.GeoDataFrame) -> xr.Dataset:
        """
        Select voxel columns at the locations of point geometries.

        Parameters
        ----------
        points : str | Path | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.

        Returns
        -------
        xr.Dataset
            Selected Xarray Dataset with dimensions "idx" and "z". The dimension "idx"
            is the index order of the selection points. The "z" dimension is the original
            depth dimension of the VoxelModel instance. The returned Dataset contains all
            original data variables of the VoxelModel instance.

        Examples
        --------
        To sample a VoxelModel with a GeoDataFrame containing point geometries:

        >>> selected = voxelmodel.select_with_points(point_gdf)

        This way, it is easily possible to sample a VoxelModel at point locations using
        GeoST Header or Collection objects by accessing their "gdf" attributes.

        Using a Header object:

        >>> selected = voxelmodel.select_with_points(Header.gdf)

        Using a Collection object:

        >>> selected = voxelmodel.select_with_points(Collection.header.gdf)

        """
        points = check_geometry_instance(points)

        if "x" in points.columns and "y" in points.columns:
            coords = points[["x", "y"]].values
        else:
            x, y = points["geometry"].x, points["geometry"].y
            coords = np.c_[x, y]

        return sample_with_coords(self.ds, coords)

    def select_with_line(self):  # pragma: no cover
        raise NotImplementedError()

    def select_within_polygons(self):  # pragma: no cover
        raise NotImplementedError()

    def select_within_bbox(self):  # pragma: no cover
        raise NotImplementedError()

    def select_by_values(self):  # pragma: no cover
        raise NotImplementedError()

    def select_top(self):  # pragma: no cover
        raise NotImplementedError()

    def select_bottom(self):  # pragma: no cover
        raise NotImplementedError()

    def slice_depth_interval(self):  # pragma: no cover
        raise NotImplementedError()

    def select_surface_level(self):  # pragma: no cover
        raise NotImplementedError()

    def zslice_to_tiff(self):  # pragma: no cover
        raise NotImplementedError()

    def get_thickness(
        self,
        condition: xr.Dataset,
        depth_range: tuple[float, float] = None,
    ) -> xr.DataArray:
        """
        Generate a thickness array of voxels in the VoxelModel that meet one or more conditions. For example,
        determine the thickness of a specific lithology within a stratigraphic unit (see example usage below).

        Parameters
        ----------
        condition : xr.DataArray
            Boolean DataArray containing that evaluate to True for the desired condition.
            For example: `voxelmodel["lith"] == 1`.
        depth_range : tuple[float, float], optional
            Search for the condition with a specified depth range of the `VoxelModel`. This should be a tuple
            containing the minimum and maximum depth values (in this order!). The default
            is None, which means the entire depth range of the VoxelModel will be used.

        Returns
        -------
        xr.DataArray
            A DataArray containing the generated map based on the specified conditions.
            The DataArray will have dimensions "y" and "x". The values in the DataArray
            represent the thickness of the selected data variable at each location.

        Examples
        --------
        Determine the thickness where the lithology in the `VoxelModel` is equal to 1:

        >>> lith_map = voxelmodel.get_thickness(voxelmodel["lithology"] == 1)

        Or generate an array on more conditions:

        >>> thickness = voxelmodel.get_thickness((voxelmodel["lithology"] == 1) & (voxelmodel["strat"] == 1100))

        Search for the condition or conditions within a specific depth window:

        >>> thickness = voxelmodel.get_thickness((voxelmodel["lithology"] == 1) & (voxelmodel["strat"] == 1100), depth_range=(-10, -20))
        """
        if depth_range:
            zmin, zmax = depth_range
            condition = condition.sel(z=slice(zmin, zmax))

        # Calculate thickness: sum non-NaN values along the z-axis and multiply by dz
        thickness = xr.where(condition, self.resolution[-1], np.nan).sum(dim="z")

        return thickness

    def to_pyvista_grid(
        self, data_vars: str | list[str] = None, structured: bool = True
    ):
        """
        Convert the VoxelModel to a PyVista grid.

        Parameters
        ----------
        data_vars : str | list[str], optional
            String representing one data variable or list of data variables to include
            in the PyVista grid. If None, all data variables are included. The default
            is None.
        structured : bool, optional
            If True, convert to a structured grid. If False, convert to an unstructured
            grid. The default is True.

        Returns
        -------
        pyvista.UnstructuredGrid or pyvista.StructuredGrid
            PyVista grid representation of the VoxelModel.
        """
        if data_vars is None:
            data_vars = self.ds.data_vars
        elif isinstance(data_vars, str):
            data_vars = [data_vars]

        if structured:
            return vtk.voxelmodel_to_pyvista_structured(
                self.ds,
                self.resolution,
                displayed_variables=data_vars,
            )
        else:
            return vtk.voxelmodel_to_pyvista_unstructured(
                self.ds,
                self.resolution,
                displayed_variables=data_vars,
            )


class LayerModel(AbstractSpatial, AbstractModel3D):  # pragma: no cover TODO: add to doc
    def __init__(self):
        raise NotImplementedError("No support of LayerModel yet.")

    @property
    def xmin(self):
        raise NotImplementedError()

    @property
    def xmax(self):
        raise NotImplementedError()

    @property
    def ymin(self):
        raise NotImplementedError()

    @property
    def ymax(self):
        raise NotImplementedError()

    @property
    def bounds(self):
        raise NotImplementedError()

    @property
    def resolution(self):
        raise NotImplementedError()

    @property
    def crs(self):
        raise NotImplementedError()

    def sel(self):
        raise NotImplementedError()

    def isel(self):
        raise NotImplementedError()

    def select_with_line(self):
        raise NotImplementedError()

    def select_with_points(self):
        raise NotImplementedError()

    def select_within_polygons(self):
        raise NotImplementedError()

    def select_within_bbox(self):
        raise NotImplementedError()

    def select_by_values(self):
        raise NotImplementedError()

    @property
    def zmin(self):
        raise NotImplementedError()

    @property
    def zmax(self):
        raise NotImplementedError()

    def select_top(self):
        raise NotImplementedError()

    def select_bottom(self):
        raise NotImplementedError()

    def slice_depth_interval(self):
        raise NotImplementedError()

    def select_surface_level(self):
        raise NotImplementedError()

    def zslice_to_tiff(self):
        raise NotImplementedError()
