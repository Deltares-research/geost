from abc import ABC, abstractmethod

import numpy as np
import rioxarray as rio
import xarray as xr


class AbstractSpatial(ABC):
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


class AbstractModel3D(ABC):
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
    def crs(self):
        return self.ds.rio.crs

    def _get_internal_zbounds(self):
        self._dz = np.abs(np.diff(self["z"])[0])
        self._zmin = np.min(self["z"].values)
        self._zmax = np.max(self["z"].values)
        self._zmin -= 0.5 * self._dz
        self._zmax += 0.5 * self._dz

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


class LayerModel(AbstractSpatial, AbstractModel3D):
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
