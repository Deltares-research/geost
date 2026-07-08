import rioxarray  # noqa: F401, register `rio` accessor
import xarray as xr

from geost.models._core import ModelType
from geost.models.modelbase import ModelBase


@xr.register_dataset_accessor("gst")
class ModelDataset(ModelBase):
    def slice_depth_interval(
        self,
    ):  # NOTE: Method will differ between voxel and layer model
        sliced = xr.Dataset(attrs=self._obj.attrs)
        for var_ in self._obj.data_vars:
            sliced[var_] = self._obj[var_].gst.slice_depth_interval()
        raise NotImplementedError()
