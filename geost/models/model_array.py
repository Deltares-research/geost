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
    pass
