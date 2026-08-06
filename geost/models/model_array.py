import xarray as xr

from geost.models import voxelmodels
from geost.models._core import ModelType
from geost.models.modelbase import ModelBase


@xr.register_dataarray_accessor("gst")
class ModelDataArray(ModelBase):
    def most_common(self, return_thickness=False) -> xr.DataArray | xr.Dataset:
        """
        Determine the "most common" value and corresponding thickness in a voxelmodel or
        a layermodel. In a voxelmodel, this calculates the mode (most frequently occurring
        value) along the depth (z) dimension for each horizontal location (x, y) in the
        model. In a layermodel, it identifies the layer with the maximum thickness at each
        x,y-location and returns the corresponding value and most common layer name.

        Parameters
        ----------
        return_thickness : bool, optional
            If True, also return the thickness of the most common value. The default is
            False. Note that a layermodel that already has a thickness variable will
            return a duplicate thickness variable if this is set to True.

        Returns
        -------
        xr.DataArray | xr.Dataset
            - For a voxelmodel: `xarray.DataArray` (x, y) with the most common value for
            each x,y-location. If `return_thickness` is True, returns an `xarray.Dataset`
            (x, y) with data variables "most_common" and "thickness".
            - For a layermodel: `xarray.Dataset` (x, y) with a "most_common" variable that
            contains the value of the most common layer and a "most_common_layer" variable
            that contains the name of the most common layer at each x,y-location.

        """
        if self._model_type == ModelType.VOXEL:
            return voxelmodels.most_common(self._obj, return_thickness=return_thickness)

        # Layer model logic is below because it differs between Dataset and DataArray
        thickness = self.top - self.bottom
        most_common_layer = thickness.idxmax(dim=self._z)
        result = most_common_layer.to_dataset(name="most_common_layer")

        result["most_common"] = self._obj.sel({self._z: most_common_layer})
        result = result.drop_vars([self._z, self._top, self._bottom])
        # We remove unnecessary coordinates to avoid confusion that get added when selecting the most common layer

        if return_thickness:
            result["thickness"] = thickness.sel({self._z: most_common_layer})

        return result
