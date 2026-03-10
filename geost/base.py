import warnings
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely import geometry as gmt

from geost import config, utils
from geost._warnings import AlignmentWarning
from geost.abstract_classes import AbstractBase
from geost.utils.projections import (
    horizontal_reference_transformer,
    vertical_reference_transformer,
)
from geost.validation import safe_validate, schemas
from geost.validation.method_checks import _requires_depth, _requires_geometry

type Coordinate = int | float
type GeometryType = gmt.base.BaseGeometry | list[gmt.base.BaseGeometry]


class Collection(AbstractBase):
    """
    A collection combines header and data and ensures that they remain aligned when
    applying methods.

    Parameters
    ----------
    header : :class:`~geost.base.PointHeader` or :class:`~geost.base.LineHeader`
        Instance of a header class corresponding to the data.
    data : :class:`~geost.base.LayeredData` or :class:`~geost.base.DiscreteData`
        Instance of a data object corresponding to the header.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        *,
        header: gpd.GeoDataFrame = None,
        has_inclined: bool = False,
        vertical_reference: str | int | CRS = 5709,
    ):
        self.data = data

        if header is None and data is not None:
            warnings.warn("Header is None, setting the header from the given data.")
            self.header = self.data.drop_duplicates("nr").reset_index(drop=True)
        else:
            self.header = header

        self._has_inclined = has_inclined

        if vertical_reference is not None:
            vertical_reference = CRS(vertical_reference)

        self._vertical_reference = vertical_reference

    def __repr__(self):
        repr_ = (
            f"{self.__class__.__name__}:\n"
            f"  header (rows, columns): {self.header.shape}\n"
            f"  data (rows, columns): {self.data.shape}\n"
            f"  # surveys = {self.n_points}\n"
        )
        return repr_

    def __len__(self):
        return len(self.header)

    @property
    def header(self):
        """
        The collection's header.
        """
        return self._header

    @property
    def data(self):
        """
        The collection's data.
        """
        return self._data

    @property
    def n_points(self):
        """
        Number of objects in the collection.
        """
        return len(self)

    @property
    def horizontal_reference(self):
        """
        Coordinate reference system represented by an instance of pyproj.crs.CRS.
        """
        return self.header.crs if self.header_has_geometry else None

    @property
    def vertical_reference(self):
        """
        Vertical datum represented by an instance of pyproj.crs.CRS.
        """
        return self._vertical_reference

    @property
    def has_inclined(self):
        """
        Boolean indicating whether there are inclined objects within the collection
        """
        return self._has_inclined

    @property
    def header_has_geometry(self):
        return self.header._geometry_column_name is not None

    @header.setter
    def header(self, header):
        if header is not None:
            if not config.validation.SKIP:
                header = safe_validate(
                    header, schemas.pointheader
                )  # TODO: validation schema needs to be inferred
        else:
            header = gpd.GeoDataFrame()

        self.set_header(header)  # This ensures header will always be a GeoDataFrame

    def set_header(self, header: pd.DataFrame | gpd.GeoDataFrame):
        if not isinstance(header, gpd.GeoDataFrame):
            header = gpd.GeoDataFrame(header)

        if not header.empty:
            if header._geometry_column_name is None:
                warnings.warn(
                    "Setting the header without an active geometry column. Spatial methods "
                    "will not work. Use collection.set_geometry to set an active geometry "
                    "column."
                )

        self._header = header
        self.check_header_to_data_alignment()

    @data.setter
    def data(self, data):
        if data is not None:
            if config.validation.SKIP:
                data = safe_validate(
                    data, schemas.layerdata
                )  # TODO: validation schema needs to be inferred
        else:
            data = pd.DataFrame()

        self._data = data
        self.check_header_to_data_alignment()

    def add_header_column_to_data(self, column_name: str) -> None:  # No change
        """
        Add a column from the header to the data table. Useful if you e.g. add some data
        to the header table, but would like to add this to each layer (row in the data
        table) as well.

        Parameters
        ----------
        column_name : str
            Name of the column in the header table to add.

        Returns
        -------
        None
            Updates the data table in place by adding the specified column from the header
            to the data table.

        """
        self._data = self._data.merge(self.header[["nr", column_name]], on="nr")

    def get(self, selection_values: str | Iterable, column: str = "nr") -> Collection:
        """
        Select all survey data by selecting a subset of the header table of the `Collection`
        instance. Can be used to select surveys by ids directly or by information available
        in another column of the header table. Optionally uses a different column than "nr"
        (the column with survey ids).

        Parameters
        ----------
        selection_values : str | Iterable
            Survey id or ids to select or values in another column of the header table
            to select.
        column : str, optional
            If not selecting by survey ids, in which column of the header to look for the
            selection values. The default is "nr".

        Returns
        -------
        :class:`geost.base.Collection`
            New Collection instance containing only the selected surveys.

        Examples
        --------
        To select surveys with object ids "obj1" and "obj2" from the collection, use:

        >>> collection.get(["obj1", "obj2"])

        Suppose we have a collection of boreholes that we have joined with geological
        map units using the method :meth:`~geost.base.PointDataCollection.get_area_labels`,
        which is then available information in the header table. We can use:

        >>> collection.get(["unit1", "unit2"], column="geological_unit")

        which selects all surveys that are located in "unit1" and "unit2" geological map
        areas.

        """
        selected_header = self.header.gst.select_by_values(column, selection_values)
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )

        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    def change_horizontal_reference(self, to_epsg: str | int | CRS):
        """
        Change the horizontal reference (i.e. coordinate reference system, crs) of the
        collection to the given target crs.

        Parameters
        ----------
        to_epsg : str | int | CRS
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input().

        Examples
        --------
        To change the collection's current horizontal reference to WGS 84 UTM zone 31N:

        >>> self.change_horizontal_reference(32631)

        This would be the same as:

        >>> self.change_horizontal_reference("epsg:32631")

        As Pyproj is very flexible, you can even use the CRS's full official name:

        >>> self.change_horizontal_reference("WGS 84 / UTM zone 31N")
        """
        transformer = horizontal_reference_transformer(
            self.horizontal_reference, to_epsg
        )
        self.data[["x", "y"]] = self.data[["x", "y"]].astype(float)
        self.data.loc[:, "x"], self.data.loc[:, "y"] = transformer.transform(
            self.data["x"], self.data["y"]
        )
        if self.has_inclined:
            self.data[["x_bot", "y_bot"]] = self.data[["x_bot", "y_bot"]].astype(float)
            self.data.loc[:, "x_bot"], self.data.loc[:, "y_bot"] = (
                transformer.transform(self.data["x_bot"], self.data["y_bot"])
            )

        self.header.gsthd.change_horizontal_reference(to_epsg)

    def change_vertical_reference(self, to_epsg: str | int | CRS):
        """
        Change the vertical reference of the collection object's surface levels

        Parameters
        ----------
        to_epsg : str | int | CRS
            EPSG of the target vertical datum. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum.

            Some often-used vertical datums are:
            - NAP : 5709
            - MSL NL depth : 9288
            - LAT NL depth : 9287
            - Ostend height : 5710

            See epsg.io for more.

        Examples
        --------
        To change the header's current vertical reference to NAP:

        >>> self.change_vertical_reference(5709)

        This would be the same as:

        >>> self.change_vertical_reference("epsg:5709")

        As the Pyproj constructors are very flexible, you can even use the CRS's full
        official name instead of an EPSG number. E.g. for changing to NAP and the
        Belgian Ostend height vertical datums repsectively, you can use:

        >>> self.change_vertical_reference("NAP")
        >>> self.change_vertical_reference("Ostend height")

        """
        transformer = vertical_reference_transformer(
            self.horizontal_reference, self.vertical_reference, to_epsg
        )
        self.data[["surface", "end"]] = self.data[["surface", "end"]].astype(float)
        _, _, new_surface = transformer.transform(
            self.data["x"], self.data["y"], self.data["surface"]
        )
        _, _, new_end = transformer.transform(
            self.data["x"], self.data["y"], self.data["end"]
        )
        self.data.loc[:, "surface"] = new_surface
        self.data.loc[:, "end"] = new_end
        self.header.gsthd.change_vertical_reference(self.vertical_reference, to_epsg)
        self._vertical_reference = CRS(to_epsg)

    def reset_header(self):
        """
        Refresh the header based on the loaded data in case the header got messed up.
        """
        self.header = self.data.gstda.to_header(self.horizontal_reference)

    def check_header_to_data_alignment(self):
        """
        Two-way check to warn of any misalignment between the header and data
        attributes. Two way, i.e. if header includes more objects than in the data and
        if the data includes more unique objects that listed in the header.

        This check is performed everytime the object is instantiated AND if any change
        is made to either the header or data attributes (see their respective setters).

        """
        warning_given = False
        if hasattr(self, "_header") and hasattr(self, "_data"):
            # In initialization of the object, _header and _data attributes do not exist yet.
            if self.header.empty and self.data.empty:
                return

            if any(~self.header["nr"].isin(self.data["nr"].unique())):
                warnings.warn(
                    "Header covers more/other objects than present in the data table. "
                    "consider running the method 'reset_header' to update the header.",
                    category=AlignmentWarning,
                )
                warning_given = True

            if not set(self.data["nr"].unique()).issubset(set(self.header["nr"])):
                warnings.warn(
                    "Header does not cover all unique objects in data. "
                    "consider running the method 'reset_header' to update the header.",
                    category=AlignmentWarning,
                )
                warning_given = True

            if config.validation.AUTO_ALIGN and warning_given:
                self.reset_header()
                print(
                    "\nNOTE: Header has been reset to align with data because AUTO_ALIGN"
                    " is enabled in the GeoST configuration.",
                )

    @_requires_geometry
    def select_within_bbox(
        self,
        xmin: int | float,
        ymin: int | float,
        xmax: int | float,
        ymax: int | float,
        invert: bool = False,
    ):
        """
        Make a selection of the collection based on a bounding box.

        Parameters
        ----------
        xmin : int | float
            Minimum x-coordinate of the bounding box.
        ymin : int | float
            Minimum y-coordinate of the bounding box.
        xmax : int | float
            Maximum x-coordinate of the bounding box.
        ymax : int | float
            Maximum y-coordinate of the bounding box.
        invert : bool, optional
            Invert the selection, then selects all objects outside of the bounding box.
            The default is False.

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing only the surveys that are located within
            the bounding box.

        """
        selected_header = self.header.gst.select_within_bbox(
            xmin, ymin, xmax, ymax, invert=invert
        )
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    @_requires_geometry
    def select_with_points(
        self,
        points: str | Path | gpd.GeoDataFrame | GeometryType,
        max_distance: float | int,
        invert: bool = False,
    ):
        """
        Select all data that lie within a maximum distance from given point geometries.

        Parameters
        ----------
        points : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of point geometries that can be used for the selection: GeoDataFrame
            containing points or filepath to a shapefile like file, or Shapely Point,
            MultiPoint or list containing Point objects.
        max_distance : float | int
            Maximum distance from the selection points.
        invert : bool, optional
            Invert the selection, selects all data that lie outside the specified maximum
            distance from the given point geometries. The default is False.

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing only the surveys that are located within
            the specified maximum distance from the given point geometries.

        """
        selected_header = self.header.gst.select_with_points(
            points, max_distance, invert=invert
        )
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    @_requires_geometry
    def select_with_lines(
        self,
        lines: str | Path | gpd.GeoDataFrame | GeometryType,
        max_distance: float | int,
        invert: bool = False,
    ):
        """
        Select all data that lie within a maximum distance from given line geometries.

        Parameters
        ----------
        lines : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of line geometries that can be used for the selection: GeoDataFrame
            containing lines or filepath to a shapefile like file, or Shapely LineString,
            MultiLineString or list containing LineString objects.
        max_distance : float | int
            Maximum distance from the selection lines.
        invert : bool, optional
            Invert the selection, selects all data that lie outside the specified maximum
            distance from the given line geometries. The default is False.

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing only the surveys that are located within
            the specified maximum distance from the given line geometries.

        """
        selected_header = self.header.gst.select_with_lines(
            lines, max_distance, invert=invert
        )
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    @_requires_geometry
    def select_within_polygons(
        self,
        polygons: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int = 0,
        invert: bool = False,
    ):
        """
        Select all data that lie within given polygon geometries.

        Parameters
        ----------
        polygons : str | Path | gpd.GeoDataFrame | PolygonGeometries
            Any type of polygon geometries that can be used for the selection: GeoDataFrame
            containing polygons or filepath to a shapefile like file, or Shapely Polygon,
            MultiPolygon or list containing Polygon objects.
        buffer : float | int, optional
            Optional buffer distance around the polygon selection geometries. The default
            is 0.
        invert : bool, optional
            Invert the selection, selects all data that lie outside the selection polygons.

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing only the surveys that are located within
            the given polygon geometries.

        """
        selected_header = self.header.gst.select_within_polygons(
            polygons, buffer=buffer, invert=invert
        )
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    @_requires_geometry
    def spatial_join(
        self,
        geometries: str | Path | gpd.GeoDataFrame,
        label_id: str | list[str],
        drop_label_if_exists: bool = True,
        include_in_header: bool = False,
        **kwargs,
    ) -> gpd.GeoDataFrame | None:
        """
        Join information from another GeoDataFrame by a spatial relationship (e.g. overlap)
        between the geometries in the header table of the Collection with the geometries in
        the other GeoDataFrame.

        Parameters
        ----------
        geometries : str | Path | gpd.GeoDataFrame
            Geometries to join with the information from. Can be a GeoDataFrame or a
            file path to a geospatial file that can be read as a GeoDataFrame.
        label_id : str | Iterable
            Column name(s) in the geometries GeoDataFrame to join the information from.
        drop_label_if_exists : bool, optional
            If True, will drop the specified 'label_id' from the original GeoDataFrame if
            these already exist as a column or columns, before the spatial join. Otherwise,
            suffixes will automatically be added to the joined columns to avoid naming conflicts.
            The default is False.
        **kwargs
            Keyword arguments to be passed to the GeoPandas :meth:`geopandas.GeoDataFrame.sjoin`
            function. See relevant documentation for more information. Note that the "how"
            parameter is not allowed when `include_in_header` is True (error is raised), as
            the spatial join will always be performed with "how='left'" in that case.

        Returns
        -------
        gpd.GeoDataFrame | None
            GeoDataFrame resulting from the spatial join if `include_in_header` is False.
            Otherwise, returns `None` and the result is included in the header of the
            Collection instance.

        """
        if include_in_header:
            if "how" in kwargs:
                raise ValueError(
                    "The 'how' parameter is not allowed when include_in_header is True."
                )
            self.header = self.header.gst.spatial_join(
                geometries,
                label_id,
                drop_label_if_exists=drop_label_if_exists,
                how="left",
                **kwargs,
            )
        else:
            return self.header.gst.spatial_join(
                geometries,
                label_id,
                drop_label_if_exists=drop_label_if_exists,
                **kwargs,
            )

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
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        """
        selected_header = self.header.gst.select_by_elevation(
            top_min=top_min, top_max=top_max, end_min=end_min, end_max=end_max
        )
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

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
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.
        """
        selected_header = self.header.gst.select_by_length(
            min_length=min_length, max_length=max_length
        )
        selected_data = self.data.gst.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    def select_by_values(
        self,
        column: str,
        values: str | Iterable | slice,
        how: str = "or",
        invert: bool = False,
        inclusive: str = "both",
    ):
        """
        Select data based on the presence of values in a given column. Can be used for
        example to select boreholes that contain peat in the lithology column.

        Parameters
        ----------
        column : str
            Name of the column to look for the selection values in.
        values : str | Iterable | slice
            Value or array-like set of values to look for in the column. In case of numerical
            values, a slice can be used to select data that contain a specific range of values,
            see example below.
        how : {"and", "or"}, optional
            Either "and" or "or", only used when multiple categorical values are provided.
            "and" requires all selection values to be present in the column for selection. "or"
            will select the data if any one of the values are found in the column. The default
            is "and".
        invert : bool, optional
            If True, the selection is inverted so that data that does not match the selection
            criteria is returned instead. The default is False.
        inclusive : {"both", "neither", "left", "right"}, optional
            When a slice is used for selection to include boundaries or not. The
            default is "both".

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing the all data from the selection result. The
            data table contains the entire data for the selected surveys, so not only the
            rows that match the selection. If that is desired, use the method
            :meth:`~geost.base.Collection.slice_by_values` instead.

        Examples
        --------
        To select data where both clay ("K") and peat ("V") are present at the same
        time, use "and" as a selection method:

        >>> data.select_by_values("lith", ["V", "K"], how="and")

        To select data that can have one, or both lithologies, use or as the selection
        method:

        >>> data.select_by_values("lith", ["V", "K"], how="or")

        In case of numerical values, use a slice to select data that contain a specific
        range of values. For example, to select data that contain cone resistances ("qc")
        between 15 and 20 MPa:

        >>> data.select_by_values("qc", slice(15, 20))

        """
        selected_data = self.data.gst.select_by_values(
            column, values, how=how, invert=invert, inclusive=inclusive
        )
        selected_header = self.header.gst.select_by_values(
            "nr", selected_data["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    @_requires_depth
    def slice_depth_interval(
        self,
        upper_boundary: float | int = None,
        lower_boundary: float | int = None,
        relative_to_vertical_reference: bool = False,
        update_layer_boundaries: bool = True,
    ):
        """
        Slice data based on given upper and lower boundaries. This returns a new object
        containing only the sliced data.

        Parameters
        ----------
        upper_boundary : float | int, optional
            Every layer that starts above this is removed. The default is None.
        lower_boundary : float | int, optional
            Every layer that starts below this is removed. The default is None.
        relative_to_vertical_reference : bool, optional
            If True, the slicing is done with respect to any kind of vertical reference
            plane (e.g. "NAP", "TAW"). If False, the slice is done with respect to depth
            below the surface. The default is False.
        update_layer_boundaries : bool, optional
            If True, the layer boundaries in the sliced data are updated according to the
            upper and lower boundaries used with the slice. If False, the original layer
            boundaries are kept in the sliced object. The default is False.

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing only the data within the specified depth
            boundaries.

        Examples
        --------
        Usage depends on whether the slicing is done with respect to depth below the
        surface or to a vertical reference plane.

        For example, select layers in data that are between 2 and 3 meters below the
        surface:

        >>> data.slice_depth_interval(2, 3)

        By default, the method updates the layer boundaries in sliced object according to
        the upper and lower boundaries. To suppress this behaviour use:

        >>> data.slice_depth_interval(2, 3, update_layer_boundaries=False)

        Slicing can also be done with respect to a vertical reference plane like "NAP".
        For example, to select layers in data that are between -3 and -5 m NAP, use:

        >>> data.slice_depth_interval(-3, -5, relative_to_vertical_reference=True)

        """
        selected_data = self.data.gst.slice_depth_interval(
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            relative_to_vertical_reference=relative_to_vertical_reference,
            update_layer_boundaries=update_layer_boundaries,
        )
        selected_header = self.header.gst.select_by_values(
            "nr", selected_data["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    def slice_by_values(
        self,
        column: str,
        values: str | Iterable | slice,
        invert: bool = False,
        inclusive: str = "both",
    ):
        """
        Slice rows from data based on matching condition. E.g. only return rows with
        a certain lithology in the collection object.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        values : str | Iterable | slice
            Value or array-like set of values to look for in the column. In case of numerical
            values, a slice can be used to slice a range of values, see example below.
        invert : bool, optional
            If True, invert the slicing action, so remove layers with selected values
            instead of keeping them. The default is False.
        inclusive : {"both", "neither", "left", "right"}, optional
            When a slice is used for selection to include boundaries or not. The
            default is "both".

        Returns
        -------
        :class:`~geost.base.Collection`
            New Collection instance containing only the data that match the slicing condition
            in the data table. The data table contains only the rows that match the slicing
            condition, so not the entire data for the selected surveys. If you want to select
            entire surveys based on the presence of certain values, use the method
            :meth:`~geost.base.Collection.select_by_values` instead.

        Examples
        --------
        Return only rows in borehole data contain sand ("Z") as lithology:

        >>> boreholes.slice_by_values("lith", "Z")

        If you want all the rows that may contain everything but sand, use the "invert"
        option:

        >>> boreholes.slice_by_values("lith", "Z", invert=True)

        In case of numerical values, use a slice object to slice data that contain a specific
        range of values. For example, to slice data that contain cone resistances ("qc")
        between 15 and 20 MPa:

        >>> data.slice_by_values("qc", slice(15, 20))

        """
        selected_data = self.data.gst.slice_by_values(
            column, values, invert=invert, inclusive=inclusive
        )
        selected_header = self.header.gst.select_by_values(
            "nr", selected_data["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    def select_by_condition(self, condition: Any, invert: bool = False):
        """
        Do a condition-based selection on the data table of the `Collection: return the
        rows in the data where the 'condition' evaluates to True, see examples below.

        Parameters
        ----------
        condition : list, pd.Series or array like
            Boolean array like object with locations at which the values will be
            preserved, dtype must be 'bool' and the length must correspond with the
            length of the data.
        invert : bool, optional
            If True, the selection is inverted so rows that evaluate to False will be
            returned. The default is False.

        Returns
        -------
        :class:`~geost.base.Collection`
            New instance containing only the rows obtained by the selection in the data
            table.

        Examples
        --------
        Select rows in data that contain a specific value:

        >>> data.select_by_condition(data["lith"] == "V")

        Select rows in the data that contain a specific (part of) string or strings:

        >>> boreholes.select_by_condition(boreholes["column"].str.contains("foo|bar"))

        Select rows in data where column values are larger than:

        >>> data.select_by_condition(data["column"] > 2)

        Or select rows in the data based on multiple conditions:

        >>> data.select_by_condition((data["column1"] > 2) & (data["column2] < 1))

        """
        selected_data = self.data.gst.select_by_condition(condition, invert)
        selected_header = self.header.gst.select_by_values(
            "nr", selected_data["nr"].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_reference=self.vertical_reference,
        )

    def get_cumulative_thickness(
        self, column: str, values: str | list[str], include_in_header: bool = False
    ) -> pd.Series | None:
        """
        Get the cumulative thickness of layers where a column contains a specified search
        value or values.

        Parameters
        ----------
        column : str
            Name of column that must contain the search value or values.
        values : str | list[str]
            Search value or values in the column to find the cumulative thickness for.
        include_in_header : bool, optional
            If True, include the result in the header table of the `Collection`. In this
            case, the method does not return anything (i.e. `None`). If False, a `pandas.Series`
            is returned. The default is False.

        Returns
        -------
        pd.Series or None
            Borehole ids and cumulative thickness of selected layers if the "include_in_header"
            option is set to False.

        Examples
        --------
        Get the cumulative thickness of the layers with lithology "K" in the column "lith"
        use:

        >>> th = boreholes.get_cumulative_thickness("lith", "K") # Returns a pandas Series

        Or get the cumulative thickness for multiple selection values. In this case, a
        Pandas DataFrame is returned with a column per selection value containing the
        cumulative thicknesses:

        >>> th = boreholes.get_cumulative_thickness("lith", ["K", "Z"]) # Returns a pandas DataFrame

        To include the result in the header object of the collection, use the
        "include_in_header" option:

        >>> boreholes.get_cumulative_thickness("lith", ["K"], include_in_header=True) # Modifies the header and returns None

        """
        thickness = self.data.gst.get_cumulative_thickness(column, values)

        if include_in_header:
            prefix = f"{column}[" if isinstance(values, slice) else ""
            suffix = "]_thickness" if isinstance(values, slice) else "_thickness"
            column_name = utils.columns.column_name_from(
                values, prefix=prefix, suffix=suffix
            )

            thickness.name = column_name
            self.header.drop(columns=column_name, errors="ignore", inplace=True)
            self.header = self.header.merge(thickness, on="nr", how="left")
        else:
            return thickness

    def get_layer_top(
        self,
        column: str,
        values: str | list[str],
        min_thickness: float = None,
        min_fraction: float = None,
        include_in_header: bool = False,
    ) -> pd.Series | None:
        """
        Find the depth at which a specified layer first occurs, starting at min_depth
        and looking downwards until the first layer of min_thickness is found of the
        specified layer.

        Parameters
        ----------
        column : str
            Name of the column to search for the specified value or values.
        values : int | float | str | list[str] | slice
            Value or values to search for in the specified column. If a slice is provided, the
            function will search for values within the specified range.
        min_thickness : float, optional
            Minimum thickness of the layer to consider. Layers thinner than this value will be
            ignored. The thickness of a layer is calculated as the difference uppermost top
            and the lowermost bottom of consecutive elements that meet the value criteria. If
            None, no minimum thickness is applied which returns the first encountered layer.
        min_fraction : float, optional
            Whether or not to allow for disturbing layers: layers that do not meet the value
            criteria in between. The minimum fraction is the minimal fraction of the 'min_thickness'
            that must meet the value criteria. If None, the entire layer must meet the criteria.
            Note that 'min_fraction' is only applied when 'min_thickness' is specified.
        include_in_header : bool, optional
            If True, include the result in the header table of the `Collection`. In this
            case, the method does not return anything (i.e. `None`). If False, a `pandas.Series`
            is returned. The default is False.

        Returns
        -------
        pd.Series or None
            Borehole ids and top levels of selected layers in meters below the surface.

        Examples
        --------
        Get the top depth of layers in boreholes where the lithology in the "lith" column
        is sand ("Z"):

        >>> tops = boreholes.get_layer_top("lith", "Z") # Returns a pandas Series

        To include the result in the header object of the collection, use the
        "include_in_header" option:

        >>> boreholes.get_layer_top("lith", "Z", include_in_header=True) # Modifies the header and returns None

        """
        top = self.data.gst.get_layer_top(
            column, values, min_thickness=min_thickness, min_fraction=min_fraction
        )

        if include_in_header:
            prefix = f"{column}[" if isinstance(values, slice) else ""
            suffix = "]_top" if isinstance(values, slice) else "_top"
            column_name = utils.columns.column_name_from(
                values, prefix=prefix, suffix=suffix
            )

            top.name = column_name
            self.header.drop(columns=column_name, errors="ignore", inplace=True)
            self.header = self.header.merge(top, on="nr", how="left")
        else:
            return top

    def get_layer_base(
        self,
        column: str,
        values: str | list[str],
        min_thickness: float = 0,
        include_in_header: bool = False,
    ):
        """
        Find the depth at which a specified layer occurs last, starting at min_depth
        and looking downwards until the first layer of min_thickness is found of the
        specified layer.

        Parameters
        ----------
        column : str
            Name of the column to search for the specified value or values.
        values : int | float | str | list[str] | slice
            Value or values to search for in the specified column. If a slice is provided, the
            function will search for values within the specified range.
        min_thickness : float, optional
            Minimum thickness of the layer to consider. Layers thinner than this value will be
            ignored. The thickness of a layer is calculated as the difference uppermost top
            and the lowermost bottom of consecutive elements that meet the value criteria. If
            None, no minimum thickness is applied which returns the first encountered layer.

        Returns
        -------
        pd.DataFrame
            Borehole ids and base levels of selected layers in meters below the surface.

        Examples
        --------
        Get the base depth of layers in boreholes where the lithology in the "lith" column
        is sand ("Z"):

        >>> base = boreholes.get_layer_base("lith", "Z") # Returns a pandas Series

        To include the result in the header object of the collection, use the
        "include_in_header" option:

        >>> boreholes.get_layer_base("lith", "Z", include_in_header=True) # Modifies the header and returns None

        """
        base = self.data.gst.get_layer_base(column, values, min_thickness=min_thickness)

        if include_in_header:
            prefix = f"{column}[" if isinstance(values, slice) else ""
            suffix = "]_base" if isinstance(values, slice) else "_base"
            column_name = utils.columns.column_name_from(
                values, prefix=prefix, suffix=suffix
            )
            base.name = column_name
            self.header.drop(columns=column_name, errors="ignore", inplace=True)
            self.header = self.header.merge(base, on="nr", how="left")
        else:
            return base

    def to_geoparquet(self, outfile: str | Path, **kwargs):
        """
        Write header data to geoparquet. You can use the resulting file to display
        borehole locations in GIS for instance. Please note that Geoparquet is supported
        by GDAL >= 3.5. For Qgis this means QGis >= 3.26

        Parameters
        ----------
        file : str | Path
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_parquet kwargs. See relevant Pandas documentation.

        """
        self.header.to_parquet(outfile, **kwargs)

    def to_shape(self, outfile: str | Path, **kwargs):
        """
        Write header data to shapefile. You can use the resulting file to display
        borehole locations in GIS for instance.

        Parameters
        ----------
        file : str | Path
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_file kwargs. See relevant GeoPandas documentation.

        """
        self.header.to_file(outfile, **kwargs)

    def to_geopackage(self, outfile: str | Path, metadata: dict[str, str] = None):
        """
        Write header and data to a geopackage. You can use the resulting file to display
        borehole locations in GIS for instance.

        Parameters
        ----------
        file : str | Path
            Path to geopackage to be written.
        metadata : dict[str, str], default None
            Optional metadata to be stored in the file. Keys and values must be strings.
            The default is None.

        """
        utils.io_helpers._to_geopackage(
            self.header,
            outfile,
            "Header",
            layer="header",
            driver="GPKG",
            index=False,
            metadata=metadata,
        )
        utils.io_helpers._to_geopackage(
            gpd.GeoDataFrame(self.data),
            outfile,
            "Data",
            layer="data",
            driver="GPKG",
            index=False,
            metadata=metadata,
        )

    def to_parquet(self, outfile: str | Path, data_table: bool = True, **kwargs):
        """
        Export the data or header table to a parquet file. By default the data table is
        exported.

        Parameters
        ----------
        file : str | Path
            Path to parquet file to be written.
        data_table : bool, optional
            If True, the data table is exported. If False, the header table is exported.
        **kwargs
            pd.DataFrame.to_parquet kwargs. See relevant Pandas documentation.

        Examples
        --------
        Export the data table:
        >>> collection.to_parquet("example.parquet")

        Export the header table:
        >>> collection.to_parquet("example.parquet", data_table=False)

        """
        if data_table:
            self.data.to_parquet(outfile, **kwargs)
        else:
            self.header.to_parquet(outfile, **kwargs)

    def to_csv(self, outfile: str | Path, data_table: bool = True, **kwargs):
        """
        Export the data or header table to a csv file. By default the data table is
        exported.

        Parameters
        ----------
        file : str | Path
            Path to csv file to be written.
        data_table : bool, optional
            If True, the data table is exported. If False, the header table is exported.
        **kwargs
            pd.DataFrame.to_csv kwargs. See relevant Pandas documentation.

        Examples
        --------
        Export the data table:
        >>> collection.to_csv("example.csv")

        Export the header table:
        >>> collection.to_csv("example.csv", data_table=False)

        """
        if data_table:
            self.data.to_csv(outfile, **kwargs)
        else:
            self.header.to_csv(outfile, **kwargs)

    def to_pickle(self, outfile: str | Path, **kwargs):
        """
        Export the complete Collection to a pickle file.

        Parameters
        ----------
        file : str | Path
            Path to pickle file to be written.
        **kwargs
            pd.to_pickle kwargs. See relevant Pandas documentation.

        Examples
        --------
        >>> collection.to_pickle("example.pkl")

        """
        utils.io_helpers.save_pickle(self, outfile, **kwargs)

    def to_pyvista_cylinders(
        self,
        displayed_variables: str | list[str],
        radius: float = 1,
        vertical_factor: float = 1.0,
        relative_to_vertical_reference: bool = True,
    ):
        """
        Create a Pyvista MultiBlock object of cylinder-shaped geometries to represent
        boreholes. Although cylinders are prettier when visualized, they are quite costly
        to render in large numbers. Consider using
        :meth:`~geost.base.Collection.to_pyvista_grid` instead for large datasets.

        Parameters
        ----------
        displayed_variables : str | list[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m in the MultiBlock. The default is 1.
        vertical_factor : float, optional
            Factor to correct vertical scale. For example, when layer boundaries are given
            in cm, use 0.01 to convert to m. The default is 1.0, so no correction is applied.
            It is not recommended to use this for vertical exaggeration, use viewer functionality
            for that instead.
        relative_to_vertical_reference : bool, optional
            If True, the depth of the objects in the vtm file will be with respect to a
            reference plane (e.g. "NAP", "TAW"). If False, the depth will be with respect
            to 0.0. The default is True.

        Returns
        -------
        pyvista.MultiBlock
            A composite class holding the data which can be iterated over.

        """
        return self.data.gstda.to_pyvista_cylinders(
            displayed_variables, radius, vertical_factor, relative_to_vertical_reference
        )

    def to_pyvista_grid(
        self,
        displayed_variables: str | list[str],
        radius: float = 1.0,
    ):
        """
        Create a Pyvista UnstructuredGrid object to represent boreholes. This is more efficient
        than :meth:`~geost.base.Collection.to_pyvista_cylinders` for large datasets, but
        less visually appealing.

        Parameters
        ----------
        displayed_variables : str | list[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m in the MultiBlock. The default is 1.

        Returns
        -------
        pyvista.UniformGrid
            A grid class holding the data which can be iterated over.

        """
        return self.data.gstda.to_pyvista_grid(displayed_variables, radius)

    def to_datafusiontools(
        self,
        columns: list[str],
        outfile: str | Path = None,
        encode: bool = False,
        relative_to_vertical_reference: bool = True,
    ):
        """
        Export all data to the core "Data" class of Deltares DataFusionTools. Returns
        a list of "Data" objects, one for each data object that is exported. This list
        can directly be used within DataFusionTools. If out_file is given, the list of
        Data objects is saved to a pickle file.

        For DataFusionTools visit:
        https://bitbucket.org/DeltaresGEO/datafusiontools/src/master/

        Parameters
        ----------
        columns : list[str]
            Which columns in the data to include for the export. These will become variables
            in the DataFusionTools "Data" class.
        outfile : str | Path, optional
            If a path to outfile is given, the data is written to a pickle file.
        encode : bool, default True
            If True, categorical data columns are encoded to additional binary columns
            (all possible values become a seperate feature that is 0 or 1). The default is
            False. Warning: if there is a large number of possible categories, many columns
            with categorical data or both, the export process may become slow and may consume
            a large amount memory. Please consider carefully which categorical data columns
            need to be included.
        relative_to_vertical_reference : bool, optional
            If True, the depth of all data objects will converted to a depth with respect to
            a reference plane (e.g. "NAP", "Ostend height"). If False, the depth will be
            kept as original in the "top" and "bottom" columns which is in meter below
            the surface. The default is True.

        Returns
        -------
        list[Data]
            List containing the DataFusionTools Data objects.

        """
        return self.data.gstda.to_datafusiontools(
            columns, outfile, encode, relative_to_vertical_reference
        )


class BoreholeCollection(Collection):
    """
    A collection combines header and borehole data and ensures that they remain aligned
    when applying methods.

    Parameters
    ----------
    header : :class:`~geost.base.PointHeader`
        Instance of a header class corresponding to the data.
    data : :class:`~geost.base.LayeredData`
        Instance of a data object corresponding to the header.
    """

    def add_grainsize_data(self, sample_data: pd.DataFrame):
        """
        Add grain size data to the borehole collection.

        Parameters
        ----------
        sample_data : pd.DataFrame
            DataFrame containing sample data with a column "nr" that contains borehole ids.

        """
        # safe_validate(DataSchemas.grainsize_data, sample_data, inplace=True)
        # sample_data["nr"].unique()
        # warnings.warn(
        #     "Header covers more/other objects than present in the data table, "
        #     "consider running the method 'reset_header' to update the header.",
        #     AlignmentWarning,
        # )
        # self.sample_data = sample_data
        raise NotImplementedError

    def to_qgis3d(
        self,
        outfile: str | Path,
        relative_to_vertical_reference: bool = True,
        **kwargs,
    ):
        """
        Write data to geopackage file that can be directly loaded in the Qgis2threejs
        plugin. Works only for layered (borehole) data.

        Parameters
        ----------
        outfile : str | Path
            Path to geopackage file to be written.
        relative_to_vertical_reference : bool, optional
            If True, the depth of all data objects will converted to a depth with respect to
            a reference plane (e.g. "NAP", "TAW"). If False, the depth will be kept as original
            in the "top" and "bottom" columns which is in meter below the surface. The default
            is True.

        **kwargs
            geopandas.GeodataFrame.to_file kwargs. See relevant Geopandas documentation.

        """
        self.data.gstda.to_qgis3d(
            outfile,
            relative_to_vertical_reference,
            crs=self.horizontal_reference,
            **kwargs,
        )

    def to_kingdom(
        self,
        outfile: str | Path,
        tdstart: int = 1,
        vw: float = 1500.0,
        vs: float = 1600.0,
    ):
        """
        Write data to 2 csv files: interval data and time-depth chart,
            for import in Kingdom seismic interpretation software.

        Parameters
        ----------
        out_file : str | Path
            Path to csv file to be written.
        tdstart : int
            startindex for TDchart, default is 1
        vw : float
            sound velocity in water in m/s, default is 1500 m/s
        vs : float
            sound velocity in sediment in m/s, default is 1600 m/s
        """
        self.data.gstda.to_kingdom(outfile, tdstart, vw, vs)


class CptCollection(Collection):
    @property  # NOTE: Temporary fix to use correct schema for validation.
    def data(self):
        """
        The collection's data.
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = safe_validate(
            data, schemas.discretedata
        )  # TODO: validation schema needs to be inferred
        self.check_header_to_data_alignment()

    def slice_depth_interval(
        self,
        upper_boundary: float | int = None,
        lower_boundary: float | int = None,
        relative_to_vertical_reference: bool = False,
    ):
        """
        Slice data based on given upper and lower boundaries. This returns a new object
        containing only the sliced data.

        Parameters
        ----------
        upper_boundary : float | int, optional
            Every layer that starts above this is removed. The default is None.
        lower_boundary : float | int, optional
            Every layer that starts below this is removed. The default is None.
        relative_to_vertical_reference : bool, optional
            If True, the slicing is done with respect to any kind of vertical reference
            plane (e.g. "NAP", "TAW"). If False, the slice is done with respect to depth
            below the surface. The default is False.

        Returns
        -------
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        Examples
        --------
        Usage depends on whether the slicing is done with respect to depth below the
        surface or to a vertical reference plane.

        For example, select layers in data that are between 2 and 3 meters below the
        surface:

        >>> data.slice_depth_interval(2, 3)

        By default, the method updates the layer boundaries in sliced object according to
        the upper and lower boundaries. To suppress this behaviour use:

        >>> data.slice_depth_interval(2, 3, update_layer_boundaries=False)

        Slicing can also be done with respect to a vertical reference plane like "NAP".
        For example, to select layers in data that are between -3 and -5 m NAP, use:

        >>> data.slice_depth_interval(-3, -5, relative_to_vertical_reference=True)

        """
        selected_data = self.data.gstda.slice_depth_interval(
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            relative_to_vertical_reference=relative_to_vertical_reference,
        )
        selected_header = self.header.gsthd.get(selected_data["nr"].unique())
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def get_cumulative_thickness(self):  # pragma: no cover
        raise NotImplementedError()

    def get_layer_top(self):  # pragma: no cover
        raise NotImplementedError()
