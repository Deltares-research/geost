from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import rioxarray  # noqa: F401, register `rio` accessor
import xarray as xr

from geost.exceptions import InvalidModelError, MissingCRSError, ModelTypeError
from geost.models import layermodels, voxelmodels
from geost.models._core import ModelType, get_model_specs
from geost.utils import conversion
from geost.utils.spatial import get_points_along_lines

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry.base import BaseGeometry

type GeometryType = BaseGeometry | list[BaseGeometry]


class ModelBase:
    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
        self._obj = xarray_obj
        self._x: str = None
        self._y: str = None
        self._z: str = None
        self._model_type: ModelType = None
        self._top: str = None
        self._bottom: str = None

        model_spec = get_model_specs(xarray_obj)
        if model_spec is not None:
            self._x = model_spec.x_dim
            self._y = model_spec.y_dim
            self._z = model_spec.z_dim
            self._model_type: ModelType = model_spec.model_type
            self._top = model_spec.top
            self._bottom = model_spec.bottom

        self._validate_model()

        # Initialize properties for caching
        self._surface_level = None
        self._zmin = None
        self._zmax = None

    def _validate_model(self):
        errors = []
        if not self._has_xy():
            errors.append("Missing x and/or y dimensions.")
        if not self._has_z():
            errors.append(
                "Missing z dimension for voxelmodel or top/bottom for layermodel."
            )
        if errors:
            raise InvalidModelError("Invalid model: \n" + "\n".join(errors))

    def _has_xy(self) -> bool:
        return self._x is not None and self._y is not None

    def _has_z(self) -> bool:
        if self._model_type == ModelType.VOXEL:
            return self._z is not None
        elif self._model_type == ModelType.LAYER:
            return self._top is not None and self._bottom is not None
        return False

    @property
    def crs(self) -> CRS | None:
        """
        The Coordinate Reference System (CRS) represented as a `pyproj.CRS` object.

        """
        return self._obj.rio.crs

    @property
    def ndims(self) -> int:
        """
        The number of dimensions in the `Dataset`/`DataArray`

        """
        return len(self._obj.dims)

    @property
    def x_dim(self) -> str:
        """
        The name of the x dimension in the `Dataset`/`DataArray`

        """
        return self._x

    @property
    def y_dim(self) -> str:
        """
        The name of the y dimension in the `Dataset`/`DataArray`

        """
        return self._y

    @property
    def z_dim(self) -> str:
        """
        The name of the z dimension in the `Dataset`/`DataArray`

        """
        return self._z

    @property
    def x(self) -> xr.DataArray:
        """
        The x coordinates of the model as an `xarray.DataArray`.

        """
        return self._obj[self._x]

    @property
    def y(self) -> xr.DataArray:
        """
        The y coordinates of the model as an `xarray.DataArray`.

        """
        return self._obj[self._y]

    @property
    def z(self) -> xr.DataArray:
        """
        The coordinates of the model's vertical dimension as an `xarray.DataArray`.

        """
        return self._obj[self._z]

    @property
    def top(self) -> xr.DataArray:
        """
        `xarray.DataArray` with the depths of the top of each layer in a layermodel.

        """
        if self._model_type != ModelType.LAYER:
            raise ModelTypeError("Only ModelType.LAYER has a 'top' property.")
        return self._obj[self._top]

    @property
    def bottom(self) -> xr.DataArray:
        """
        `xarray.DataArray` with the depths of the bottom of each layer in a layermodel.

        """
        if self._model_type != ModelType.LAYER:
            raise ModelTypeError("Only ModelType.LAYER has a 'bottom' property.")
        return self._obj[self._bottom]

    @property
    def surface_level(self) -> xr.DataArray:
        """
        `xarray.DataArray` with the surface level elevation of the model.

        """
        if self._surface_level is not None:
            return self._surface_level

        if self._model_type == ModelType.VOXEL:
            self._surface_level = voxelmodels.get_surface_level(self._obj)
        elif self._model_type == ModelType.LAYER:
            self._surface_level = self.top.max(dim=self._z)

        return self._surface_level

    @property
    def model_type(self) -> ModelType:
        """
        The type of the model as a `ModelType` enum member.

        """
        return self._model_type

    def resolution(
        self, meters: bool = False
    ) -> tuple[float, float] | tuple[float, float, float]:
        """
        Determine the resolution of the model.

        Returns
        -------
        tuple[float, float] | tuple[float, float, float]
            Resolution of the model. For a voxelmodel, returns (xres, yres, zres). For a
            layermodel, returns (xres, yres).
        meters : bool, optional
            If True, the resolution is returned in meters. If False, the resolution is
            returned in the units of the model's CRS. The default is False.

        Raises
        ------
        ValueError
            Resolution cannot be determined for 1D models.

        """
        try:
            grid = self._obj.isel({self._z: 0})
            if self.crs.is_geographic and meters:
                grid = grid.rio.reproject(grid.rio.estimate_utm_crs())
            xres, yres = grid.rio.resolution()
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
            raise TypeError(
                "Vertical bounds cannot be determined for a layermodel DataArray. This method "
                "can only be used on an xarray.Dataset with valid 'top' and 'bottom' variables."
            )

        bottom, top = self._internal_z_bounds()

        if self._model_type == ModelType.VOXEL:
            _, _, resolution_z = self.resolution()
            top += 0.5 * resolution_z
            bottom -= 0.5 * resolution_z

        return bottom, top

    @property
    def shape(self):
        """
        The shape of the model as a tuple of integers with the size of each dimension.

        """
        return tuple(self._obj.sizes.values())

    def write_crs(self, crs: str | int | CRS, inplace: bool = False, **kwargs) -> None:
        """
        Write the coordinate reference system (CRS) to the model. This method only writes
        a valid CRS to the model and does not perform any reprojection of the data. See
        Rioxarray documentation for reprojection methods.

        Parameters
        ----------
        crs : str | int | CRS
            The coordinate reference system to write to the model. This can be a string
            (e.g., "EPSG:4326"), an integer (e.g., 4326), or a pyproj.CRS object.
        inplace : bool, optional
            If True, the CRS is written to the model in place. If False, a new model with
            the CRS written is returned. The default is False.
        **kwargs
            Additional keyword arguments to pass to the `rio.write_crs` method. See
            relevant documentation for details.

        """
        return self._obj.rio.write_crs(crs, inplace=inplace, **kwargs)

    def slice_xy(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> xr.Dataset | xr.DataArray:
        """
        Slice the model in the x and y dimensions.

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate of the slice.
        ymin : float
            Minimum y-coordinate of the slice.
        xmax : float
            Maximum x-coordinate of the slice.
        ymax : float
            Maximum y-coordinate of the slice.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Sliced model in the x and y dimensions.

        """
        if self._obj.indexes[self._x].is_monotonic_decreasing:
            xmin, xmax = xmax, xmin
        if self._obj.indexes[self._y].is_monotonic_decreasing:
            ymin, ymax = ymax, ymin
        return self._obj.sel({self._x: slice(xmin, xmax), self._y: slice(ymin, ymax)})

    def select_within_bbox(
        self,
        xmin: int | float,
        ymin: int | float,
        xmax: int | float,
        ymax: int | float,
        crs: str | int | CRS | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """
        Select data within a specified bounding box (xmin, ymin, xmax, ymax). This method
        needs the model to have a valid coordinate reference system (CRS) defined.

        Parameters
        ----------
        xmin : float | int
            Minimum x-coordinate of the bounding box.
        ymin : float | int
            Minimum y-coordinate of the bounding box.
        xmax : float | int
            Maximum x-coordinate of the bounding box.
        ymax : float | int
            Maximum y-coordinate of the bounding box.
        crs : str | int | CRS | None, optional
            Coordinate reference system of the bounding box. If None, the CRS of the
            bounding box is assumed to be the same as the dataset.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Subset of the original dataset or data array within the specified bounding box.

        Raises
        ------
        ValueError
            If no data is found within the specified bounding box.

        """
        try:
            return self._obj.rio.clip_box(
                minx=xmin,
                miny=ymin,
                maxx=xmax,
                maxy=ymax,
                crs=crs,
                allow_one_dimensional_raster=True,  # Otherwise rioxarray raises an error
            )
        except rioxarray.exceptions.NoDataInBounds as e:
            raise ValueError(
                "No data found within the specified bounding box: "
                f"({xmin}, {ymin}, {xmax}, {ymax})"
            ) from e
        except rioxarray.exceptions.MissingCRS as e:
            raise MissingCRSError(
                "Missing CRS for the dataset, use `model.gst.write_crs(crs)` to set "
                "the CRS before selecting within a bounding box."
            ) from e

    def select_points(
        self,
        points: str | Path | gpd.GeoDataFrame | GeometryType,
        crs: str | int | CRS | None = None,
        drop: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """
        Select model data at specified point locations. The points can be provided as a
        GeoDataFrame, a shapely geometry, or a path to a file that can be read into a
        GeoDataFrame. The points must have a valid coordinate reference system (CRS) that
        matches the model's CRS, or the user can specify the CRS of the points using the
        `crs` parameter.

        Parameters
        ----------
        points : str | Path | gpd.GeoDataFrame | GeometryType
            Points to select.
        crs : str | int | CRS | None, optional
            Coordinate reference system of the points. If None, the CRS of the
            points is assumed to be the same as the model. The default is None.
        drop : bool, optional
            If True, points outside the model bounds are removed from the result. If
            False, points outside the model bounds result in full NaN columns. The
            default is True.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Subset of the original Dataset or DataArray with coordinate "idx" corresponding
            to the selected points.

        Example
        -------
        Use a GeoDataFrame with points to select the model at the x,y-locations of the points:

        >>> import geopandas as gpd
        >>> points = gpd.GeoDataFrame(
        ...     geometry=gpd.points_from_xy([0.8, 2.4, 1.0], [0.8, 2.4, 0.5]), crs="EPSG:28992"
        ... )
        >>> model.gst.select_points(points) # Select the model at the point locations

        If the points are in a different CRS than the model, specify the CRS of the points:

        >>> points_wgs = points.to_crs(4326) # Change the CRS of the points to WGS84
        >>> model.gst.select_points(points_wgs, crs=4326) # Specify the CRS of the points

        """
        points = conversion.check_geometry_instance(points)

        if crs is not None and crs != self.crs:
            points = points.to_crs(self.crs)

        xmin, ymin, xmax, ymax = self.bounds()
        points_in_bounds = points.cx[xmin:xmax, ymin:ymax]

        sel = self._obj.sel(
            x=xr.DataArray(points_in_bounds.geometry.x, dims="idx"),
            y=xr.DataArray(points_in_bounds.geometry.y, dims="idx"),
            method="nearest",
        )

        sel = sel.assign_coords(idx=("idx", points_in_bounds.index))

        if not drop:
            sel = sel.reindex(idx=points.index)

        return sel

    def select_along_lines(
        self,
        lines: str | Path | gpd.GeoDataFrame | GeometryType,
        crs: str | int | CRS | None = None,
        distance: float | int = None,
        start_at_zero: bool = True,
        drop: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """
        Select model data at specified distances along given line geometries. The lines
        can be provided as a GeoDataFrame, a shapely geometry, or a path to a file that
        can be read into a GeoDataFrame. The lines must have a valid coordinate reference
        system (CRS) that matches the model's CRS, or the user can specify the CRS of the
        lines using the `crs` parameter.

        Parameters
        ----------
        lines : str | Path | gpd.GeoDataFrame | GeometryType
            Lines to select along.
        crs : str | int | CRS | None, optional
            Coordinate reference system of the lines. If None, the CRS of the
            lines is assumed to be the same as the model. The default is None.
        distance : float | int, optional
            Distance between points along the lines in meters. If None, the distance is
            set to the model's x-resolution in meters. The distance will be used to compute
            the evenly spaced sampling points along each line. For each line, the model
            is sampled at the computed sampling points. The default is None.
        start_at_zero : bool, optional
            If True, the first point along each line is at distance 0. If False, the first
            point is at half the distance. The default is True.
        drop : bool, optional
            If True, anything that falls outside the model extent is dropped. So complete
            lines outside the extent can be dropped or the parts of the lines outside the
            extent can be dropped. The default is True.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Subset of the original Dataset or DataArray with coordinates "line" and
            "distance" corresponding to the selected lines and distances along the lines.

        Example
        -------
        Use a GeoDataFrame with lines to select the model at the x,y-locations along the
        lines:

        >>> import geopandas as gpd
        >>> lines = gpd.GeoDataFrame(
        ...     geometry=[
        ...         shapely.LineString([(0.8, 0.9), (2.4, 2.5)]),
        ...         shapely.LineString([(1.0, 0.5), (2.0, 1.5)]),
        ...     ],
        ...     crs="EPSG:28992"
        ... )
        >>> model.gst.select_along_lines(lines, distance=0.2) # Select the model along the lines at 10m intervals

        If the points are in a different CRS than the model, specify the CRS of the points:

        >>> lines_wgs = lines.to_crs(4326) # Change the CRS of the lines to WGS84
        >>> model.gst.select_along_lines(lines_wgs, crs=4326) # Specify the CRS of the lines

        """
        if distance is None:
            xres, *_ = self.resolution(meters=True)
            distance = abs(xres)

        lines = conversion.check_geometry_instance(lines)
        if lines.crs is None and crs is None:
            lines = lines.set_crs(self.crs)
        elif crs is not None and crs != self.crs:
            lines = lines.to_crs(self.crs)

        # We need all distance measures in meters so if geographic CRS, convert units to meters
        if lines.crs.is_geographic:
            lines_utm = lines.to_crs(lines.estimate_utm_crs())
        else:
            lines_utm = lines

        # Add a small fraction of the distance to ensure the last point is included in the range
        max_line_length = lines_utm.length.max() + 0.1 * distance
        if start_at_zero:
            distance = np.arange(0, max_line_length, distance)
        else:
            distance = np.arange(0.5 * distance, max_line_length, distance)

        xmin, ymin, xmax, ymax = self.bounds()
        lines_in_bounds = lines.cx[xmin:xmax, ymin:ymax]

        # Get the points to sample from the lines at each distance: CRS is in meters
        sample_points = get_points_along_lines(
            lines_utm.loc[lines_in_bounds.index], distance
        )
        sample_points.set_index("distance", append=True, inplace=True)

        if sample_points.crs != self.crs:
            sample_points.to_crs(self.crs, inplace=True)

        points_in_bounds = sample_points.cx[xmin:xmax, ymin:ymax]

        sel = self._obj.sel(
            x=xr.DataArray(points_in_bounds.geometry.x, dims="point"),
            y=xr.DataArray(points_in_bounds.geometry.y, dims="point"),
            method="nearest",
        )

        # Create dimensions ("z/layer", "line", "distance")
        coords = xr.Coordinates.from_pandas_multiindex(points_in_bounds.index, "point")
        sel = sel.assign_coords(coords).unstack("point")

        if not drop:
            sel = sel.reindex(line=lines.index, distance=distance)

        return sel

    def mask_geometries(
        self,
        geometries: str | Path | gpd.GeoDataFrame | GeometryType,
        crs: str | int | CRS = None,
        all_touched: bool = False,
        invert: bool = False,
        drop: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """
        Mask 'x','y'-locations that overlap with geometries (points, lines, polygons).

        Parameters
        ----------
        geometries : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of geometry that can be used to mask the model. This can be a path
            to shapefile-like file, a GeoDataFrame, or a single geometry or list of geometries.
        crs : str | int | CRS, optional
            The CRS of the input geometries. The default is None, then it is assumed to be
            the same as the dataset's CRS.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be selected. If false, only
            pixels whose center is within the polygon or that are selected by Bresenham's
            line algorithm (in case of lines) will selected. In case the geometries are
            any cell that overlaps with a geometry will be selected. The default is False.
        invert : bool, optional
            If True, the mask will be inverted. By default False.
        drop : bool, optional
            If True, drop the data outside of the extent of the mask geometries. Otherwise,
            it will return the result with the original shape. The default is False.

        Returns
        -------
        xr.Dataset | xr.DataArray
            The masked xarray.Dataset or xarray.DataArray.

        Example
        -------
        Use a GeoDataFrame with points to select the x,y-locations of the model that overlap
        with the points:

        >>> import geopandas as gpd
        >>> points = gpd.GeoDataFrame(
        ...     geometry=gpd.points_from_xy([0.8, 2.4, 1.0], [0.8, 2.4, 0.5]), crs="EPSG:28992"
        ... )
        >>> model.gst.mask_geometries(points, drop=False) # Mask the point locations and keep original shape

        Or use a Shapely geometry to select the overlapping x,y-locations of the model:

        >>> import shapely
        >>> line = shapely.LineString([(0.8, 0.9), (2.4, 2.5)])
        >>> model.gst.mask_geometries(line, crs=28992) # Specify the CRS of the line

        """
        geometries = conversion.check_geometry_instance(geometries)

        clipped = self._obj
        # Ensure that the dimensions are in the correct order
        if self.ndims == 3 and clipped.dims != (self._y, self._x, self._z):
            clipped = clipped.transpose(self._y, self._x, self._z)
        elif self.ndims == 2 and clipped.dims != (self._y, self._x):
            clipped = clipped.transpose(self._y, self._x)  # Also in 2D

        return clipped.rio.clip(
            geometries.geometry.values,
            crs=crs,
            all_touched=all_touched,
            invert=invert,
            drop=drop,
        )

    def slice_depth_interval(  # TODO: add slice with respect to surface level
        self,
        upper: int | float | np.ndarray | xr.DataArray = None,
        lower: int | float | np.ndarray | xr.DataArray = None,
        how: Literal["overlap", "majority", "inner"] = "overlap",
        update_top_bottom: bool = True,
        drop: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """
        Slice a specified depth interval from a voxelmodel or layermodel between upper
        and lower bounds.

        Parameters
        ----------
        upper, lower : int | float | xr.DataArray, optional
            Upper and/or lower bound of the depth interval. This can be a single value or
            a 1D or 2D DataArray containing variable depths. In case of a DataArray, a 1D
            DataArray should contain either the "x" or "y" dimension and a 2D DataArray
            should contain both "x" and "y" dimensions. Otherwise broadcasting cannot be
            done correctly and the slicing cannot be done. The default is None.
        how : {"overlap", "majority", "inner"}, optional
            Method to use for slicing. This parameter is only applicable to voxelmodels
            (i.e., `model.gst.model_type` is `ModelType.VOXEL`) and will be ignored for
            layermodels. The default is "overlap".
            - "overlap": Include voxels that at least partially overlap with the specified
            depth interval.
            - "majority": Include voxels that have 50% or more of their volume within
            the specified depth interval.
            - "inner": Include only voxels that are completely within the specified depth
            interval.
        update_top_bottom : bool, optional
            If True, the "top" and "bottom" variables of a layermodel will be updated to
            reflect the new depth interval after slicing. If False, the "top" and "bottom"
            variables will remain unchanged. This parameter is only applicable to layermodels
            (i.e., `model.gst.model_type` is `ModelType.LAYER`) and will be ignored for
            voxelmodels. The default is True.
        drop : bool, optional
            If True, depths where the result only contains missing values will be dropped
            from the slice result. If False, the original shape is kept. The default is
            True.

        Returns
        -------
        xr.Dataset | xr.DataArray
            A new xarray.Dataset or xarray.DataArray containing only the data within the
            specified depth interval.

        Examples
        --------
        Slice a fixed depth interval between -10 and -20:

        >>> sliced = model.gst.slice_depth_interval(upper=-10, lower=-20)

        Slice a model of 2 rows and 2 columns between variable depth intervals using
        DataArrays.

        >>> upper = xr.DataArray([[-10, -15], [-12, -18]], dims=("y", "x"))
        >>> upper  # 2D array with different depth interval for each "x" and "y"
        <xarray.DataArray (y: 2, x: 2)>
        array([[-15, -20],
               [-17, -23]])
        Dimensions without coordinates: y, x
        >>> sliced = model.gst.slice_depth_interval(upper=upper, lower=upper - 5)

        >>> upper = xr.DataArray([-10, -15], dims=("y",)) # slice along "y"
        >>> upper  # 1D array with different depth interval for each "y" but the same for every "x"
        <xarray.DataArray (y: 2)>
        array([-15, -20])
        Dimensions without coordinates: y
        >>> sliced = model.gst.slice_depth_interval(upper=upper, lower=upper - 5)

        """
        if self._model_type == ModelType.VOXEL:
            return voxelmodels.slice_depth_interval(
                self._obj, upper=upper, lower=lower, how=how, drop=drop
            )
        elif self._model_type == ModelType.LAYER:
            return layermodels.slice_depth_interval(
                self._obj,
                upper=upper,
                lower=lower,
                update_top_bottom=update_top_bottom,
                drop=drop,
            )

    def get_thickness(self, condition: xr.DataArray | xr.Dataset) -> xr.DataArray:
        """
        Calculate the thickness of a voxelmodel or layermodel based on a specified
        condition. The condition can be a boolean DataArray or Dataset that indicates
        which voxels or layers to include in the thickness calculation.

        Parameters
        ----------
        condition : xr.DataArray | xr.Dataset
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

        Or in a voxelmodel `DataArray` of stratigraphy the thickness where the stratigraphy
        equals 1100 or 1200:

        >>> thickness = voxelmodel_strat.gst.get_thickness(
        ...    (voxelmodel_strat == 1100) | (voxelmodel_strat == 1200)
        ... )

        Or in a layermodel for a subset of units where the a value is smaller than 10:

        >>> thickness = layermodel.gst.get_thickness(
        ...    (layermodel["layer"].isin(["B", "D"])) & (layermodel["value"] < 10)
        ... )

        """
        condition, _ = xr.broadcast(condition, self._obj)

        if self._model_type == ModelType.VOXEL:
            *_, zres = self.resolution()
            thickness = xr.where(condition, zres, 0)
        elif self._model_type == ModelType.LAYER:
            thickness = self.top - self.bottom
            thickness = xr.where(condition, thickness, 0)

        return thickness.sum(dim=self._z)
