from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely import geometry as gmt

from geost import spatial, utils
from geost.abstract_classes import AbstractHeader
from geost.mixins import GeopandasExportMixin
from geost.projections import vertical_reference_transformer
from geost.validation import safe_validate, schemas

type Coordinate = int | float
type GeometryType = gmt.base.BaseGeometry | list[gmt.base.BaseGeometry]


class PointHeader(AbstractHeader):
    def __init__(self, gdf):
        self._gdf = gdf

    # def change_horizontal_reference(self, to_epsg: str | int | CRS):
    #     """
    #     Change the horizontal reference (i.e. coordinate reference system, crs) of the
    #     header to the given target crs.

    #     Parameters
    #     ----------
    #     to_epsg : str | int | CRS
    #         EPSG of the target crs. Takes anything that can be interpreted by
    #         pyproj.crs.CRS.from_user_input().

    #     Examples
    #     --------
    #     To change the header's current horizontal reference to WGS 84 UTM zone 31N:

    #     >>> self.change_horizontal_reference(32631)

    #     This would be the same as:

    #     >>> self.change_horizontal_reference("epsg:32631")

    #     As Pyproj is very flexible, you can even use the CRS's full official name:

    #     >>> self.change_horizontal_reference("WGS 84 / UTM zone 31N")

    #     """
    #     self._gdf = self._gdf.to_crs(to_epsg)
    #     self._gdf[["x", "y"]] = self._gdf[["x", "y"]].astype(float)
    #     self._gdf["x"] = self._gdf["geometry"].x
    #     self._gdf["y"] = self._gdf["geometry"].y

    # def change_vertical_reference(self, to_epsg: str | int | CRS):
    #     """
    #     Change the vertical reference of the object's surface levels.

    #     Parameters
    #     ----------
    #     to_epsg : str | int | CRS
    #         EPSG of the target vertical datum. Takes anything that can be interpreted by
    #         pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum.

    #         Some often-used vertical datums are:
    #         - NAP : 5709
    #         - MSL NL depth : 9288
    #         - LAT NL depth : 9287
    #         - Ostend height : 5710

    #         See epsg.io for more.

    #     Examples
    #     --------
    #     To change the header's current vertical reference to NAP:

    #     >>> self.change_horizontal_reference(5709)

    #     This would be the same as:

    #     >>> self.change_horizontal_reference("epsg:5709")

    #     As the Pyproj constructors are very flexible, you can even use the CRS's full
    #     official name instead of an EPSG number. E.g. for changing to NAP and the
    #     Belgian Ostend height vertical datums repsectively, you can use:

    #     >>> self.change_horizontal_reference("NAP")
    #     >>> self.change_horizontal_reference("Ostend height")
    #     """
    #     transformer = vertical_reference_transformer(
    #         self.horizontal_reference, self.vertical_reference, to_epsg
    #     )
    #     self.gdf[["surface", "end"]] = self.gdf[["surface", "end"]].astype(float)
    #     _, _, new_surface = transformer.transform(
    #         self.gdf["x"], self.gdf["y"], self.gdf["surface"]
    #     )
    #     _, _, new_end = transformer.transform(
    #         self.gdf["x"], self.gdf["y"], self.gdf["end"]
    #     )
    #     self._gdf.loc[:, "surface"] = new_surface
    #     self._gdf.loc[:, "end"] = new_end

    def get(self, selection_values: str | Iterable, column: str = "nr"):
        """
        Get a subset of a header through a string or iterable of object id(s).
        Optionally uses a different column than "nr" (the column with object ids).

        Parameters
        ----------
        selection_values : str | Iterable
            Values to select.
        column : str, optional
            In which column of the header to look for selection values, by default "nr".

        Returns
        -------
        Instance of :class:`~geost.base.PointHeader`.
            Instance of :class:`~geost.base.PointHeader` containing only
            objects selected through this method.

        Examples
        --------
        >>> self.get(["obj1", "obj2"])

        will return a collection with only these objects.

        Suppose we have a number of boreholes that we have joined with geological
        map units using the method
        :meth:`~geost.base.PointHeader.get_area_labels`. We have added this data
        to the header table in the column 'geological_unit'. Using:

        >>> self.get(["unit1", "unit2"], column="geological_unit")

        will return a :class:`~geost.base.PointHeader` with all boreholes
        that are located in "unit1" and "unit2" geological map areas.

        """
        if isinstance(selection_values, str):
            selection = self._gdf[self._gdf[column] == selection_values]
        elif isinstance(selection_values, Iterable):
            selection = self._gdf[self._gdf[column].isin(selection_values)]

        selection = selection[~selection.duplicated()]

        return selection

    def select_within_bbox(
        self,
        xmin: Coordinate,
        ymin: Coordinate,
        xmax: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on a bounding box.

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
        invert : bool, optional
            Invert the selection, so select all objects outside of the
            bounding box in this case, by default False.

        Returns
        -------
        :class:`~geost.base.PointHeader`
            Instance of :class:`~geost.base.PointHeader` containing only selected
            geometries.
        """
        selection = spatial.select_points_within_bbox(
            self._gdf, xmin, ymin, xmax, ymax, invert=invert
        )
        return selection

    def select_with_points(
        self,
        points: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on point geometries.

        Parameters
        ----------
        points : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of point geometries that can be used for the selection: GeoDataFrame
            containing points or filepath to a shapefile like file, or Shapely Point,
            MultiPoint or list containing Point objects.
        buffer : float | int
            Buffer distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.base.PointHeader`
            Instance of :class:`~geost.base.PointHeader` containing only selected
            geometries.

        """
        selection = spatial.select_points_near_points(
            self.gdf, points, buffer, invert=invert
        )
        return selection

    def select_with_lines(
        self,
        lines: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on line geometries.

        Parameters
        ----------
        lines : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of line geometries that can be used for the selection: GeoDataFrame
            containing lines or filepath to a shapefile like file, or Shapely LineString,
            MultiLineString or list containing LineString objects.
        buffer : float | int
            Buffer distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.base.PointHeader`
            Instance of :class:`~geost.base.PointHeader` containing only selected
            geometries.

        """
        selection = spatial.select_points_near_lines(
            self._gdf, lines, buffer, invert=invert
        )
        return selection

    def select_within_polygons(
        self,
        polygons: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int = 0,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on polygon geometries.

        Parameters
        ----------
        polygons : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of polygon geometries that can be used for the selection: GeoDataFrame
            containing polygons or filepath to a shapefile like file, or Shapely Polygon,
            MultiPolygon or list containing Polygon objects.
        buffer : float | int, optional
            Optional buffer distance around the polygon selection geometries, by default
            0.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.base.PointHeader`
            Instance of :class:`~geost.base.PointHeader` containing only selected
            geometries.

        """
        selection = spatial.select_points_within_polygons(
            self._gdf, polygons, buffer, invert=invert
        )
        return selection

    def select_by_depth(
        self,
        top_min: float = None,
        top_max: float = None,
        end_min: float = None,
        end_max: float = None,
    ):
        """
        Select data from depth constraints. If a keyword argument is not given it will
        not be considered. e.g. if you need only boreholes that go deeper than -500 m
        use only end_max = -500.

        Parameters
        ----------
        top_min : float, optional
            Minimum elevation of the borehole/cpt top, by default None.
        top_max : float, optional
            Maximum elevation of the borehole/cpt top, by default None.
        end_min : float, optional
            Minimum elevation of the borehole/cpt end, by default None.
        end_max : float, optional
            Maximumelevation of the borehole/cpt end, by default None.

        Returns
        -------
        Child of :class:`~geost.base.PointHeader`.
            Instance of :class:`~geost.base.PointHeader` or containing only objects
            selected by this method.

        """
        selected = self._gdf.copy()
        if top_min is not None:
            selected = selected[selected["surface"] >= top_min]
        if top_max is not None:
            selected = selected[selected["surface"] <= top_max]
        if end_min is not None:
            selected = selected[selected["end"] >= end_min]
        if end_max is not None:
            selected = selected[selected["end"] <= end_max]

        selected = selected[~selected.duplicated()]

        return selected

    def select_by_length(self, min_length: float = None, max_length: float = None):
        """
        Select data from length constraints: e.g. all boreholes between 50 and 150 m
        long. If a keyword argument is not given it will not be considered.

        Parameters
        ----------
        min_length : float, optional
            Minimum length of borehole/cpt, by default None.
        max_length : float, optional
            Maximum length of borehole/cpt, by default None.

        Returns
        -------
        Child of :class:`~geost.base.PointHeader`.
            Instance of :class:`~geost.base.PointHeader` or containing only objects
            selected by this method.

        """
        selected = self._gdf.copy()
        length = selected["surface"] - selected["end"]
        if min_length is not None:
            selected = selected[length >= min_length]
        if max_length is not None:
            selected = selected[length <= max_length]

        selected = selected[~selected.duplicated()]

        return selected

    def get_area_labels(
        self,
        polygon_gdf: str | Path | gpd.GeoDataFrame,
        column_name: str | Iterable,
        include_in_header=False,
    ) -> pd.DataFrame:
        """
        Find in which area (polygons) the point data locations fall. e.g. to determine
        in which geomorphological unit points are located.

        Parameters
        ----------
        polygon_gdf : str | Path | gpd.GeoDataFrame
            GeoDataFrame with polygons.
        column_name : str | Iterable
            The column name to find the labels in. Given as a string or iterable of
            strings in case you'd like to find multiple labels.
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default
            False.

        Returns
        -------
        pd.DataFrame
            Borehole ids and the polygon label they are in. If include_in_header = True,
            a column containing the generated data will be added inplace.
        """
        polygon_gdf = utils.check_geometry_instance(polygon_gdf)
        polygon_gdf = spatial.check_and_coerce_crs(polygon_gdf, self._gdf.crs)

        all_nrs = self._gdf["nr"]
        area_labels = spatial.find_area_labels(self._gdf, polygon_gdf, column_name)
        area_labels = pd.concat([all_nrs, area_labels], axis=1)

        if include_in_header:
            self._gdf.drop(
                columns=column_name,
                errors="ignore",
                inplace=True,
            )
            self._gdf = self._gdf.merge(area_labels, on="nr")
        else:
            return area_labels


class LineHeader(AbstractHeader):  # pragma: no cover
    def __init__(self, gdf):
        self._gdf = gdf

    def change_horizontal_reference(self, to_epsg: str | int | CRS):
        raise NotImplementedError("Add function logic")

    def change_vertical_reference(self, to_epsg: str | int | CRS):
        raise NotImplementedError("Add function logic")

    def get(self):
        raise NotImplementedError("Add function logic")

    def select_within_bbox(self):
        raise NotImplementedError("Add function logic")

    def select_with_points(self):
        raise NotImplementedError("Add function logic")

    def select_with_lines(self):
        raise NotImplementedError("Add function logic")

    def select_within_polygons(self):
        raise NotImplementedError("Add function logic")

    def select_by_depth(self):
        raise NotImplementedError("Add function logic")

    def select_by_length(self):
        raise NotImplementedError("Add function logic")

    def get_area_labels(self):
        raise NotImplementedError("Add function logic")
