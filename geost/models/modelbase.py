from enum import Enum
from typing import NamedTuple

import rioxarray  # noqa: F401, register `rio` accessor
import xarray as xr

from geost.models._core import ModelType, detect_top_and_bottom, detect_vertical_dim


class ModelBase:
    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
        self._obj = xarray_obj
        self._x: str = xarray_obj.rio._x_dim  # Utilize rioxarray if possible
        self._y: str = xarray_obj.rio._y_dim
        self._z: str = None
        self._top: str = None
        self._bottom: str = None

        vertical_spec = detect_vertical_dim(xarray_obj)
        if vertical_spec is not None:
            self._z = vertical_spec.z_dim
            self._model_type: ModelType = vertical_spec.model_type

            if self._model_type == ModelType.LAYER:
                top, bottom = detect_top_and_bottom(xarray_obj)
                self._top = top
                self._bottom = bottom

    @property
    def crs(self):
        return self._obj.rio.crs

    @property
    def x_dim(self):
        return self._x

    @property
    def y_dim(self):
        return self._y

    @property
    def z_dim(self):
        return self._z

    @property
    def model_type(self):
        return self._model_type

    def write_crs(self, crs, **kwargs):
        return self._obj.rio.write_crs(crs, **kwargs)

    def resolution(self):
        raise NotImplementedError()

    def bounds(self):  # pragma: no cover
        raise NotImplementedError()

    def vertical_bounds(self):  # pragma: no cover
        raise NotImplementedError()

    @property
    def shape(self):
        return tuple(self._obj.sizes.values())

    def select_with_points(self):  # pragma: no cover
        """
        Implementation of method does not differ between DataArray or Dataset and VoxelModel
        and LayerModel.

        """
        raise NotImplementedError()

    def select_with_line(self):  # pragma: no cover
        """
        Implementation of method does not differ between DataArray or Dataset and VoxelModel
        and LayerModel.

        """
        raise NotImplementedError()

    def select_within_polygons(self):  # pragma: no cover
        """
        Implementation of method does not differ between DataArray or Dataset and VoxelModel
        and LayerModel.

        """
        raise NotImplementedError()

    def select_within_bbox(self):  # pragma: no cover
        """
        Implementation of method does not differ between DataArray or Dataset and VoxelModel
        and LayerModel.

        """
        raise NotImplementedError()
