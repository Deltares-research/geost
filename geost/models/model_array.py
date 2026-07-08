import rioxarray  # noqa: F401, register `rio` accessor
import xarray as xr

from geost.models._core import ModelType
from geost.models.modelbase import ModelBase


@xr.register_dataarray_accessor("gst")
class ModelDataArray(ModelBase):
    def slice_depth_interval(
        self,
    ):  # NOTE: Method will differ between voxel and layer model
        if self._model_type == ModelType.VOXEL:
            print("Slicing depth interval for voxel model")
        elif self._model_type == ModelType.LAYER:
            print("Slicing depth interval for layer model")
