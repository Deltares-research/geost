import xarray as xr

from geost.bro.geotop import GeotopUnits, _evaluate_geotop_condition, _GeotopCondition
from geost.models import voxelmodels
from geost.models._core import ModelType
from geost.models.modelbase import ModelBase


@xr.register_dataset_accessor("gst")
class ModelDataset(ModelBase):
    def _evaluate_condition(
        self, condition: GeotopUnits | _GeotopCondition | xr.DataArray
    ) -> xr.DataArray:
        """
        Helper to evaluate conditions using `GeotopUnits` (and other GeoST objects in the
        future) in methods that require conditional input. This translates all input into
        valid conditional masks.

        """
        if isinstance(condition, (GeotopUnits, _GeotopCondition)):
            condition = _evaluate_geotop_condition(condition, self._obj)

        condition, _ = xr.broadcast(condition, self._obj)
        return condition

    def get_top_bottom(self, condition: GeotopUnits | xr.DataArray) -> xr.Dataset:
        """
        Derive the top and bottom depths at each x,y-location of a specified condition
        from a voxelmodel or layermodel. The condition can be a boolean DataArray or
        Dataset that indicates which voxels or layers to include in the thickness
        calculation.

        Parameters
        ----------
        condition : GeotopUnits | xr.DataArray
            A boolean DataArray or Dataset indicating which voxels or layers to include.
            The condition should have the same dimensions as the model.

        Returns
        -------
        xr.Dataset
            xarray.Dataset containing "top" and "bottom" DataArrays for each horizontal
            x,y-location in the model.

        Examples
        --------
        Determine the top and bottom in a voxelmodel `Dataset` where "strat" equals 1100 and
        "lith" equals 1:

        >>> top_bottom = voxelmodel.gst.get_top_bottom(
        ...    (voxelmodel["lithology"] == 1) & (voxelmodel["strat"] == 1100)
        ... )

        Or in a layermodel for a subset of units where the a value is smaller than 10:

        >>> top_bottom = layermodel.gst.get_top_bottom(
        ...    (layermodel["layer"].isin(["B", "D"])) & (layermodel["value"] < 10)
        ... )

        If you are working with GeoTOP, you can also use a :class:`~geost.bro.geotop.GeotopUnits`
        object to specify the condition, for example to get the top and bottom of the "Formatie van
        Echteld" unit with clay as the lithology:

        >>> geotop = geost.read_geotop_from_opendap(bbox=(110_000, 440_000, 120_000, 450_000))
        >>> strat_units = geost.bro.geotop_strat_units()
        >>> lithok_units = geost.bro.geotop_lithok_units()

        >>> top_bottom = geotop.gst.get_top_bottom(
        ...     strat_units.select_description_contains("Formatie van Echteld")
        ...     & lithok_units.select_unit("k")
        ... )

        """
        condition = self._evaluate_condition(condition)

        if self._model_type == ModelType.VOXEL:
            *_, zres = self.resolution()
            depth = self.z.where(condition)
            top = depth.max(dim=self._z) + abs(0.5 * zres)
            bottom = depth.min(dim=self._z) - abs(0.5 * zres)
        elif self._model_type == ModelType.LAYER:
            top = self.top.where(condition).max(dim=self._z)
            bottom = self.bottom.where(condition).min(dim=self._z)

        return xr.Dataset({"top": top, "bottom": bottom})

    def get_thickness(self, condition: GeotopUnits | xr.DataArray) -> xr.DataArray:
        """
        Calculate the thickness where a voxelmodel or layermodel meets a specified
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
        Determine the thickness in a voxelmodel `Dataset` where "strat" equals 1100 and
        "lith" equals 1:

        >>> thickness = voxelmodel.gst.get_thickness(
        ...    (voxelmodel["lithology"] == 1) & (voxelmodel["strat"] == 1100)
        ... )

        Or in a layermodel for a subset of units where the a value is smaller than 10:

        >>> thickness = layermodel.gst.get_thickness(
        ...    (layermodel["layer"].isin(["B", "D"])) & (layermodel["value"] < 10)
        ... )

        If you are working with GeoTOP, you can also use a :class:`~geost.bro.geotop.GeotopUnits`
        object to specify the condition, for example to get the thickness of the "Formatie van
        Echteld" unit with clay as the lithology:

        >>> geotop = geost.read_geotop_from_opendap(bbox=(110_000, 440_000, 120_000, 450_000))
        >>> strat_units = geost.bro.geotop_strat_units()
        >>> lithok_units = geost.bro.geotop_lithok_units()

        >>> thickness = geotop.gst.get_thickness(
        ...     strat_units.select_description_contains("Formatie van Echteld")
        ...     & lithok_units.select_unit("k")
        ... )

        """
        condition = self._evaluate_condition(condition)

        if self._model_type == ModelType.VOXEL:
            *_, zres = self.resolution()
            thickness = xr.where(condition, zres, 0)
        elif self._model_type == ModelType.LAYER:
            thickness = self.top - self.bottom
            thickness = xr.where(condition, thickness, 0)

        return thickness.sum(dim=self._z)

    def most_common(
        self, return_thickness=False, only_most_common_layer=False
    ) -> xr.DataArray | xr.Dataset:
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
        only_most_common_layer : bool, optional
            If True, only return the most common layer and thickness (if `return_thickness=True`)
            for layermodels. The default is False. This option is ignored for voxelmodels.

        Returns
        -------
        xr.DataArray | xr.Dataset
            - For a voxelmodel: `xarray.DataArray` (x, y, data_var) with the most common
            value for each x,y-location for each data variable. If `return_thickness`
            is True, returns an `xarray.Dataset` (x, y, data_var) with data variables
            "most_common" and "thickness".
            - For a layermodel: `xarray.Dataset` (x, y) with a "most_common" variable for
            each of the data variables and an added "most_common_layer" variable that
            contains the name of the most common layer at each x,y-location.

        """
        if self._model_type == ModelType.VOXEL:
            variables = []
            results = []
            for var in self._obj.data_vars:
                most_common = voxelmodels.most_common(self._obj[var], return_thickness)
                variables.append(var)
                results.append(most_common)

            result = xr.concat(results, dim="data_var").assign_coords(
                data_var=variables
            )
        else:
            thickness = self.top - self.bottom
            most_common_layer = thickness.idxmax(dim=self._z)
            result = most_common_layer.to_dataset(name="most_common_layer")

            if return_thickness:
                result["thickness"] = thickness.sel({self._z: most_common_layer})

            if only_most_common_layer:
                return result

            vars_3d = [
                var for var in self._obj.data_vars if self._z in self._obj[var].dims
            ]
            for var in vars_3d:
                result_var = f"{var}_most_common"
                result[result_var] = self._obj[var].sel({self._z: most_common_layer})

            result = result.drop_vars(
                [self._z, self._top, self._bottom], errors="ignore"
            )

        return result

    def to_pyvista_grid(self, structured: bool = True, **kwargs):
        raise NotImplementedError(
            "The `to_pyvista_grid` method is not implemented for ModelDataset. "
        )
