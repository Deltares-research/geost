from typing import TYPE_CHECKING, Literal

import xarray as xr

from geost.models import layermodels as lm
from geost.models import voxelmodels as vm
from geost.models._core import ModelType
from geost.models.modelbase import ModelBase

if TYPE_CHECKING:
    import numpy as np


@xr.register_dataarray_accessor("gst")
class ModelDataArray(ModelBase):
    def slice_depth_interval(
        self,
        upper: int | float | np.ndarray | xr.DataArray = None,
        lower: int | float | np.ndarray | xr.DataArray = None,
        how: Literal["overlap", "majority", "inner"] = "overlap",
        update_top_bottom: bool = True,
        drop: bool = True,
    ) -> xr.Dataset:
        if self._model_type == ModelType.VOXEL:
            return vm.slice_depth_interval(
                self._obj, upper=upper, lower=lower, how=how, drop=drop
            )
        elif self._model_type == ModelType.LAYER:
            pass

        raise NotImplementedError(
            f"Slice depth interval not implemented for model type: {self._model_type}"
        )
