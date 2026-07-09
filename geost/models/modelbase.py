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

        # Initialize properties for caching
        self._zmin = None
        self._zmax = None

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

    def resolution(self) -> tuple[float, float] | tuple[float, float, float]:
        """
        Determine the resolution of the model.

        Returns
        -------
        tuple[float, float] | tuple[float, float, float]
            Resolution of the model. For a voxelmodel, returns (xres, yres, zres). For a
            layermodel, returns (xres, yres).

        Raises
        ------
        ValueError
            Resolution cannot be determined for 1D models.

        """
        try:
            xres, yres = self._obj.rio.resolution()
        except rioxarray.exceptions.DimensionError as e:
            raise ValueError("Resolution cannot be determined for 1D models.") from e

        if self._model_type == ModelType.VOXEL:
            bottom, top = self._internal_z_bounds()
            zres = (top - bottom) / (self._obj.sizes[self._z] - 1)
            return xres, yres, zres

        return xres, yres

    def bounds(self) -> tuple[float, float, float, float]:
        """
        Determine the bounding box of the model.

        Returns
        -------
        tuple[float, float, float, float]
            Bounding box of the model (xmin, ymin, xmax, ymax).

        """
        return self._obj.rio.bounds()

    def _internal_z_bounds(self):  # pragma: no cover
        if self._zmin is not None and self._zmax is not None:
            return self._zmin, self._zmax

        if self._model_type == ModelType.VOXEL:
            top = float(self._obj[self._z].max())
            bottom = float(self._obj[self._z].min())
        elif self._model_type == ModelType.LAYER:
            top = float(self._obj[self._top].max())
            bottom = float(self._obj[self._bottom].min())

        # Cache the computed bounds for future calls
        self._zmin = bottom
        self._zmax = top

        return bottom, top

    def vertical_bounds(self) -> tuple[float, float]:
        """
        Determine the vertical bounds (zmin, zmax) of the model.

        Returns
        -------
        tuple[float, float]
            Vertical bounds of the model (zmin, zmax).

        """
        if isinstance(self._obj, xr.DataArray) and self._model_type == ModelType.LAYER:
            pass  # TODO: think about what to do when not a dataset, but a dataarray. Should we raise an error?

        bottom, top = self._internal_z_bounds()

        if self._model_type == ModelType.VOXEL:
            _, _, resolution_z = self.resolution()
            top += 0.5 * resolution_z
            bottom -= 0.5 * resolution_z

        return bottom, top

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
