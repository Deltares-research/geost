from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray as rio
import xarray as xr

from geost.spatial import check_gdf_instance

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
    def select(self):
        pass

    @abstractmethod
    def select_index(self):
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
        dy, dx = np.abs(self.ds.rio.resolution())
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

    def select(self, **xr_kwargs):
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
        >>> selected = voxelmodel.select(x=[1, 2, 3])  # Using keyword arguments
        >>> selected = voxelmodel.select({"x": [1, 2, 3]})  # Using a dictionary
        >>> selected = voxelmodel.select(x=slice(1, 4))  # Using a slice

        Using additional options as well. For instance, when the desired coordinates do
        not exactly match the VoxelModel coordinates, select the nearest:
        >>> selected = voxelmodel.select(x=[1.1, 2.5, 3.3], method="nearest")

        """
        selected = self.ds.sel(**xr_kwargs)
        return self.__class__(selected)

    def select_index(self, **xr_kwargs):
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
        >>> selected = voxelmodel.select(x=[1, 2, 3])  # Using keyword arguments
        >>> selected = voxelmodel.select({"x": [1, 2, 3]})  # Using a dictionary
        >>> selected = voxelmodel.select(x=slice(1, 4))  # Using a slice

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
        points = check_gdf_instance(points)

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


class LayerModel(AbstractSpatial, AbstractModel3D):  # pragma: no cover
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

    def select(self):
        raise NotImplementedError()

    def select_index(self):
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
