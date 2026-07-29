from typing import TYPE_CHECKING

import rioxarray  # noqa: F401, register `rio` accessor
import xarray as xr

from geost.models._core import ModelType, detect_top_and_bottom, detect_vertical_dim
from geost.utils import conversion

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry.base import BaseGeometry

type GeometryType = BaseGeometry | list[BaseGeometry]


class ModelBase:
    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray):
        self._obj = xarray_obj
        self._x: str = xarray_obj.rio._x_dim  # Utilize rioxarray if possible
        self._y: str = xarray_obj.rio._y_dim
        self._z: str = None
        self._model_type: ModelType = None
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
    def ndims(self):
        return len(self._obj.dims)

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
        return tuple(self._obj.sizes.values())

    def select_within_bbox(
        self,
        xmin: int | float,
        ymin: int | float,
        xmax: int | float,
        ymax: int | float,
        crs: str | int | CRS | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """
        Select data within a specified bounding box (xmin, ymin, xmax, ymax).

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
            points is assumed to be the same as the model.
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

        coords = points_in_bounds.get_coordinates()

        sel = self._obj.sel(
            x=xr.DataArray(coords["x"], dims="idx"),
            y=xr.DataArray(coords["y"], dims="idx"),
            method="nearest",
        )  # "x" and "y" are standard geopandas names from `get_coordinates()`

        sel = sel.assign_coords(idx=("idx", coords.index))

        if not drop:
            sel = sel.reindex(idx=points.index)

        return sel

    def select_along_line(self):  # pragma: no cover
        """
        This method is intended to select data along a line. This can be done in two ways:
        1) sample an x,y-point at each distance x along the line.
        2) take n-samples along the line between start and end.

        The result should be a new xarray.Dataset or xarray.DataArray with (distance, z)
        dimensions. The distance dimension should be the same length as the number of
        points sampled along the line. If part of the line is outside the model bounds,
        the result should contain full NaN columns for those distances or the distances
        should be removed from the result.

        """
        raise NotImplementedError()

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
