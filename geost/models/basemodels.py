from abc import ABC, abstractmethod

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
    def bounds(self):
        pass

    @property
    @abstractmethod
    def resolution(self):
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


class AbstractSpatial3D(ABC):
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


class VoxelModel(AbstractSpatial, AbstractSpatial3D):
    pass


class LayerModel(AbstractSpatial, AbstractSpatial3D):
    pass
