import warnings

import numpy as np
import xarray as xr

from geost.bro.geotop import GeotopUnits
from geost.models import voxelmodels
from geost.models._core import ModelType
from geost.models.modelbase import ModelBase


@xr.register_dataarray_accessor("gst")
class ModelDataArray(ModelBase):
    def get_thickness(self, condition: GeotopUnits | xr.DataArray) -> xr.DataArray:
        """
        Calculate the thickness of a voxelmodel or layermodel based on a specified
        condition. The condition can be a boolean DataArray or Dataset that indicates
        which voxels or layers to include in the thickness calculation.

        Parameters
        ----------
        condition : GeotopUnits | xr.DataArray
            A boolean DataArray or Dataset indicating which voxels or layers to include
            in the thickness calculation. The condition should have the same dimensions
            as the model.

        Returns
        -------
        xr.DataArray
            xarray.DataArray containing the calculated thickness for each horizontal x,y-
            location in the model.

        Examples
        --------
        Determine the thickness in a voxelmodel `DataArray` of stratigraphy where the
        stratigraphy equals 1100 or 1200:

        >>> thickness = voxelmodel_strat.gst.get_thickness(
        ...    (voxelmodel_strat == 1100) | (voxelmodel_strat == 1200)
        ... )

        Or in a layermodel for a subset of units:

        >>> thickness = layermodel.gst.get_thickness(layermodel["layer"].isin(["B", "D"]))

        If you are working with GeoTOP, you can also use a :class:`~geost.bro.geotop.GeotopUnits`
        object to specify the condition, for example to get the thickness of the "Formatie van
        Echteld" unit:

        >>> geotop = geost.read_geotop_from_opendap(bbox=(110_000, 440_000, 120_000, 450_000))
        >>> strat_units = geost.bro.geotop_strat_units()
        >>> echteld = strat_units.select_description_contains("Formatie van Echteld")
        >>> thickness_echteld = geotop.gst.get_thickness(echteld)

        """
        if isinstance(condition, GeotopUnits):
            warnings.warn(
                "The model version cannot be found in a DataArray of a GeoTOP variable, "
                "cannot check if the metadata version matches the model version. Please "
                "check the model version against the `xarray.Dataset` of GeoTOP.",
                UserWarning,
            )
            condition = self._obj.isin(condition.voxel_nr)

        condition, _ = xr.broadcast(condition, self._obj)

        if self._model_type == ModelType.VOXEL:
            *_, zres = self.resolution()
            thickness = xr.where(condition, zres, 0)
        elif self._model_type == ModelType.LAYER:
            thickness = self.top - self.bottom
            thickness = xr.where(condition, thickness, 0)

        return thickness.sum(dim=self._z)

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

    def value_counts(self, dim: str = None, normalize: bool = False) -> xr.DataArray:
        """
        Get the value counts of unique values in a DataArray.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to count unique values. The default is None.
        normalize : bool, optional
            If True, return the relative frequencies of the unique values instead of the
            absolute counts. The default is False.

        Returns
        -------
        xr.DataArray
            DataArray containing the counts of unique values along the specified dimension.

        """
        var_ = self._obj.values
        values, counts = np.unique(var_[~np.isnan(var_)], return_counts=True)

        name = self._obj.name or "variable"

        if dim is None:
            counts = xr.DataArray(counts, coords={name: values})
        else:
            counts = [(self._obj == v).sum(dim=dim) for v in values]
            counts = xr.concat(counts, dim=xr.DataArray(values, dims=name))

        if normalize:
            total = counts.sum(dim=name)
            counts = counts / total

        return counts

    def to_pyvista_grid(self, structured: bool = True, **kwargs):
        raise NotImplementedError(
            "The `to_pyvista_grid` method is not implemented for ModelDataArray. "
        )
