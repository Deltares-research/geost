import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from geost import config, utils
from geost._warnings import AlignmentWarning
from geost.abstract_classes import AbstractBase
from geost.utils.projections import (
    horizontal_reference_transformer,
    vertical_reference_transformer,
)
from geost.validation.method_checks import (
    _requires_depth,
    _requires_geometry,
    _requires_surface,
    _requires_xy,
)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


type Coordinate = int | float
type GeometryType = BaseGeometry | list[BaseGeometry]


class Collection(AbstractBase):
    """
    A `Collection` is GeoST's data container for any kind of subsurface data. It keeps
    survey data in a "header" and "data" table and ensures that they remain aligned when
    applying methods.

    - The header is a `GeoDataFrame` that contains one row per survey (e.g. borehole, CPT,
    etc.) with metadata about the survey and spatial information.
    - The data is a `DataFrame` that contains the logged information for all surveys, with
    one row per logged layer in the survey.

    Parameters
    ----------
    data : :class:`pd.DataFrame`, optional
        Data table containing the logged information for all surveys, with one row per
        logged layer in the survey. If not given, an empty DataFrame is set. The default
        is None.
    header : :class:`gpd.GeoDataFrame`, optional
        Header containing one row per survey with metadata and spatial information. If not
        given, the header will automatically be set from the data by dropping duplicates
        in the column identifying the survey (e.g. "nr") and keeping the first row for each
        survey. The default is None.
    has_inclined : bool, optional
        Boolean indicating whether there are inclined objects within the collection. This
        is used to determine whether the collection contains objects with bottom coordinates
        (e.g. "x_bot", "y_bot") in the data table. The default is False.
    vertical_reference : str | int | CRS, optional
        Vertical datum represented by an instance of pyproj.crs.CRS or anything that can be
        interpreted by pyproj.crs.CRS.from_user_input(). This is used to keep track of the
        vertical reference of the surface levels in the collection, which is important for
        any method that involves vertical reference transformations. The default is 5709.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        *,
        header: gpd.GeoDataFrame = None,
        has_inclined: bool = False,
        vertical_datum: str | int | CRS = None,
    ):
        if data is None or data.empty:
            if header is not None and not header.empty:
                raise ValueError(
                    "Header was provided but data is None. A header cannot exist "
                    "without a corresponding data table."
                )
            data = pd.DataFrame()
            header = gpd.GeoDataFrame()
            survey_id_col = None

        # data is not None in both conditions below, otherwise header and data are always empty
        elif header is None or header.empty:
            warnings.warn("Header is None, setting the header from the given data.")
            survey_id_col = self._check_survey_id_col(data, "Data")
            header = data.drop_duplicates(survey_id_col).reset_index(drop=True)
        else:
            survey_id_col_header = self._check_survey_id_col(header, "Header")
            survey_id_col = self._check_survey_id_col(data, "Data")

            if survey_id_col != survey_id_col_header:
                raise ValueError(
                    "Column identifying survey IDs in data and header must have the same name. "
                    f"Found '{survey_id_col}' in data and '{survey_id_col_header}' in header."
                )

        self._nr = survey_id_col
        self.header = header
        self.data = data

        self._vertical_datum = (
            CRS.from_user_input(vertical_datum) if vertical_datum else None
        )
        self._has_inclined = has_inclined

    def __repr__(self):
        repr_ = (
            f"{self.__class__.__name__}\n"
            f"  header (rows, columns) : {self.header.shape}\n"
            f"  data (rows, columns)   : {self.data.shape}\n"
            f"crs: {self.crs.name if self.crs else None}\n"
            f"vertical datum: {self.vertical_datum.name if self.vertical_datum else None}\n"
        )
        return repr_

    def __len__(self):
        return len(self.header)

    @staticmethod
    def _check_survey_id_col(df, df_name):
        try:
            return df.gst._nr
        except KeyError as e:
            raise KeyError(
                f"{df_name} table must contain a column identifying the survey IDs."
            ) from e

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
    def crs(self):
        """
        Coordinate reference system represented by an instance of pyproj.crs.CRS.
        """
        return self.header.crs if self.header_has_geometry else None

    @property
    def vertical_datum(self):
        """
        Vertical datum represented by an instance of pyproj.crs.CRS.
        """
        return self._vertical_datum

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
        if not data.empty:
            if not config.validation.SKIP:
                data = data.gst.validate()

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
        self._data = self._data.merge(self.header[[self._nr, column_name]], on=self._nr)

    def get(self, selection_values: str | Iterable, column: str = None) -> Collection:
        """
        Select all survey data by selecting a subset of the header table of the `Collection`
        instance. Can be used to select surveys by ids directly or by information available
        in another column of the header table. Optionally, a different column than the
        survey ID column can be used.

        Parameters
        ----------
        selection_values : str | Iterable
            Survey ID or IDs to select or values in another column of the header table
            to select.
        column : str, optional
            If not selecting by survey IDs, in which column of the header to look for the
            selection values. The default is None, then the survey ID column is used.

        Returns
        -------
        :class:`geost.base.Collection`
            New Collection instance containing only the selected surveys.

        Examples
        --------
        To select surveys with object IDs "obj1" and "obj2" from the collection, use:

        >>> collection.get(["obj1", "obj2"])

        Suppose we have a collection of boreholes that we have joined with geological
        map units using the method :meth:`~geost.base.PointDataCollection.get_area_labels`,
        which is then available information in the header table. We can use:

        >>> collection.get(["unit1", "unit2"], column="geological_unit")

        which selects all surveys that are located in "unit1" and "unit2" geological map
        areas.

        """
        column = column or self._nr
        selected_header = self.header.gst.select_by_values(column, selection_values)
        selected_data = self.data.gst.select_by_values(
            self._nr, selected_header[self._nr].unique()
        )

        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
        )

    @_requires_geometry
    def set_crs(self, crs: str | int | CRS, allow_override: bool = False) -> None:
        """
        Set the coordinate reference system (CRS) for the collection's header geometry
        column.

        Parameters
        ----------
        crs : str | int | CRS
            EPSG of the CRS to set. Takes anything that can be interpreted by
            `pyproj.crs.CRS.from_user_input()`.

        Returns
        -------
        None
            Updates the CRS of the header geometry column in place.

        """
        self.header = self.header.set_crs(
            crs, inplace=True, allow_override=allow_override
        )

    @_requires_geometry
    def to_crs(
        self,
        crs: str | int | CRS,
        xbot: str | None = None,
        ybot: str | None = None,
    ) -> None:
        """
        Change the horizontal reference (i.e. coordinate reference system, crs) of the
        collection to the given target crs.

        Parameters
        ----------
        crs : str | int | CRS
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input().
        xbot, ybot : str | None, optional
            In case data is inclined, specify labels for columns holding the coordinates
            of the bottom of each layer. These coordinates will be then transformed as well.
            The default is None, which means that these coordinates are not transformed.

        Returns
        -------
        None
            Updates the header and data of the collection in place.

        Examples
        --------
        To change the collection's current horizontal reference to WGS 84 UTM zone 31N:

        >>> collection.to_crs(32631) # Updates object in place.

        This would be the same as:

        >>> collection.to_crs("epsg:32631")

        As Pyproj is very flexible, you can even use the CRS's full official name:

        >>> collection.to_crs("WGS 84 / UTM zone 31N")

        In case of inclined layer data, make sure the bottom coordinates of each layer are
        transformed as well by specifying the column labels for these coordinates:

        >>> collection.to_crs(32631, xbot="x_bot", ybot="y_bot")

        """
        if self.crs is None:
            raise ValueError(
                "Cannot transform horizontal reference because the current CRS of the "
                "collection is not set. Use the method 'set_crs' to set the CRS first."
            )

        self.data = self.data.gst.transform_coordinates(
            self.crs, crs, xbot=xbot, ybot=ybot
        )
        self.header = self.header.gst.to_crs(crs)

    @_requires_surface
    def set_vertical_datum(self, vertical_datum: str | int | CRS) -> None:
        """
        Set the vertical datum of the collection's surface levels without performing a
        transformation of the surface levels. This can be used if you know that the
        surface levels are already in the target vertical datum, but just want to update
        the vertical datum information of the collection.

        Parameters
        ----------
        vertical_datum : str | int | CRS
            Vertical datum represented by an instance of pyproj.crs.CRS or anything that can be
            interpreted by pyproj.crs.CRS.from_user_input().

        Returns
        None

        """
        self._vertical_datum = (
            CRS.from_user_input(vertical_datum) if vertical_datum else None
        )

    @_requires_geometry
    @_requires_surface
    @_requires_xy
    def to_vertical_datum(self, vertical_datum: str | int | CRS):  # pragma: no cover
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
        To change the current vertical reference from Ostend height to NAP:

        >>> collection.change_vertical_reference(5710, 5709)

        This would be the same as:

        >>> collection.change_vertical_reference("epsg:5710", "epsg:5709")

        As the Pyproj constructors are very flexible, you can even use the CRS's full
        official name instead of an EPSG number. E.g. for changing to NAP and the
        Belgian Ostend height vertical datums repsectively, you can use:

        >>> collection.change_vertical_reference("Ostend height", "NAP")

        """
        raise NotImplementedError(
            "Custom transformation between vertical datums will be implemented in a "
            "future version of GeoST. For now, you can use the set_vertical_datum method"
            " to set the vertical datum without transforming the surface levels directly, "
            "and then perform the transformation manually"
        )

        if self.vertical_datum is None:
            raise ValueError(
                "Cannot transform vertical datum because this collection has no vertical "
                "datum set. Use the method 'set_vertical_datum' to set the vertical datum "
                "first."
            )

        data = self.data.copy()
        columns = self.data.gst.positional_columns

        self.header = self.header.gst.to_vertical_datum(vertical_datum)

        x, y = self.data[columns.get("x")], self.data[columns.get("x")]
        transformer = vertical_reference_transformer(
            self.crs, self.vertical_datum, vertical_datum
        )

        if surface := columns.get("surface"):
            _, _, new_surface = transformer.transform(x, y, self.data[surface])
            data[surface] = new_surface

        if end := columns.get("end"):
            _, _, new_end = transformer.transform(x, y, self.data[end])
            data[end] = new_end

        self.data = data
        self.set_vertical_datum(vertical_datum)

    def reset_header(self):
        """
        Refresh the header based on the loaded data in case the header got messed up.
        """
        self.header = self.data.gst.to_header(
            crs=self.crs, include_columns=list(self.header.columns)
        )

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

            if any(~self.header[self._nr].isin(self.data[self._nr].unique())):
                warnings.warn(
                    "Header covers more/other objects than present in the data table. "
                    "consider running the method 'reset_header' to update the header.",
                    category=AlignmentWarning,
                )
                warning_given = True

            if not set(self.data[self._nr].unique()).issubset(
                set(self.header[self._nr])
            ):
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
            self._nr, selected_header[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_header[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_header[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_header[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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

    @_requires_depth
    def determine_end_depth(self, survey_id_as_index: bool = True) -> pd.Series:
        """
        Determine the end depth of each survey based on the surface and depth columns in
        the data table of the Collection.

        Parameters
        ----------
        survey_id_as_index : bool, optional
            If True, the returned Series will have the survey ID column as its index. The
            default is True.

        Returns
        -------
        pd.Series
            A pandas Series containing the end depth for each survey in the data with the
            same index as the original data table.

        """
        end = self.data.gst.determine_end_depth()
        end.name = "end"
        if survey_id_as_index:
            end.index = self.data[self._nr]
        return end

    def select_by_elevation(
        self,
        top_min: float = None,
        top_max: float = None,
        end_min: float = None,
        end_max: float = None,
    ) -> Collection:
        """
        Select surveys by elevation constraints: e.g. all surveys with their surface level
        and end depths between specified minimum and maximum values. See examples below.

        Parameters
        ----------
        top_min : float, optional
            Minimum elevation of the borehole/cpt top, by default None.
        top_max : float, optional
            Maximum elevation of the borehole/cpt top, by default None.
        end_min : float, optional
            Minimum elevation of the borehole/cpt end, by default None.
        end_max : float, optional
            Maximum elevation of the borehole/cpt end, by default None.

        Returns
        -------
        :class:`~geost.base.Collection`
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        Examples
        --------
        For example, select all surveys with their surface level between 0 and 5 m and end
        depth between -20 and -10 m.

        >>> selection = collection.select_by_elevation(top_min=0, top_max=5, end_min=-20, end_max=-10)

        """
        selected_header = self.header.copy()

        if "end" not in selected_header.columns:
            ends = self.determine_end_depth(survey_id_as_index=True)
            selected_header = selected_header.merge(
                ends[~ends.index.duplicated()], on=self._nr, how="left"
            )

        selected_header = selected_header.gst.select_by_elevation(
            top_min=top_min, top_max=top_max, end_min=end_min, end_max=end_max
        )
        selected_data = self.data.gst.select_by_values(
            self._nr, selected_header[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
        )

    def select_by_length(
        self, min_length: float = None, max_length: float = None
    ) -> Collection:
        """
        Select data from length constraints: e.g. all surveys between 50 and 150 m
        long.

        Parameters
        ----------
        min_length : float, optional
            Minimum length of the survey. The default is None.
        max_length : float, optional
            Maximum length of the survey. The default is None.

        Returns
        -------
        :class:`~geost.base.Collection`
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        """
        selected_header = self.header.copy()

        if "end" not in selected_header.columns:
            ends = self.determine_end_depth(survey_id_as_index=True)
            selected_header = selected_header.merge(
                ends[~ends.index.duplicated()], on=self._nr, how="left"
            )

        selected_header = selected_header.gst.select_by_length(
            min_length=min_length, max_length=max_length
        )
        selected_data = self.data.gst.select_by_values(
            self._nr, selected_header[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_data[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_data[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_data[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self._nr, selected_data[self._nr].unique()
        )
        return self.__class__(
            selected_data,
            header=selected_header,
            has_inclined=self.has_inclined,
            vertical_datum=self.vertical_datum,
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
            self.header = self.header.merge(thickness, on=self._nr, how="left")
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
            self.header = self.header.merge(top, on=self._nr, how="left")
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
            self.header = self.header.merge(base, on=self._nr, how="left")
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
        n_sides: int = 8,
        vertical_factor: float = 1.0,
    ):
        """
        Create a Pyvista MultiBlock object of cylinder-shaped geometries to represent
        boreholes. Although cylinders are prettier when visualized, they are quite costly
        to render in large numbers. Consider using
        :meth:`~geost.base.LayeredData.to_pyvista_grid` instead for large datasets.

        Parameters
        ----------
        displayed_variables : str | List[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m in the MultiBlock. The default is 1.
        n_sides : int, optional
            Number of sides for the cylinder. The default is 8, which gives a good balance
            between visual quality and rendering performance. Increase for enhanced visual
            quality, decrease for better performance.
        vertical_factor : float, optional
            Factor to correct vertical scale. For example, when layer boundaries are given
            in cm, use 0.01 to convert to m. The default is 1.0, so no correction is applied.
            It is not recommended to use this for vertical exaggeration, use viewer functionality
            for that instead.

        Returns
        -------
        pyvista.MultiBlock
            A composite class holding the data which can be iterated over.

        """
        return self.data.gst.to_pyvista_cylinders(
            displayed_variables, radius, n_sides, vertical_factor
        )

    def to_pyvista_grid(
        self,
        displayed_variables: str | list[str],
        radius: float = 1,
    ):
        """
        Create a PyVista UnstructuredGrid object of the data in this instance. This
        method is more efficient than :meth:`~geost.base.LayeredData.to_pyvista_cylinders`
        for large datasets, as it uses a grid representation instead of cylinders.

        Parameters
        ----------
        displayed_variables : str | list[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float
            The 'radius' of the voxels. This will determine the
            horizontal size of the voxels in the resulting unstructured grid.

        Returns
        -------
        pyvista.UnstructuredGrid
            A PyVista UnstructuredGrid object containing the data that can be used for
            3D visualisation in PyVista or other VTK viewers.

        """
        return self.data.gst.to_pyvista_grid(displayed_variables, radius)

    def to_qgis3d(
        self,
        outfile: str | Path,
        crs: str | CRS = None,
        **kwargs,
    ):
        """
        Write data to geopackage file that can be directly loaded in the Qgis2threejs
        plugin. Works only for layered (borehole) data.

        Parameters
        ----------
        outfile : str | Path
            Path to geopackage file to be written.
        crs : str | CRS, optional
            Coordinate reference system to use for the geometries in the output file. If
            None, the horizontal reference of the Collection is used. The default is None.

        **kwargs
            geopandas.GeodataFrame.to_file kwargs. See relevant Geopandas documentation.

        """
        crs = CRS(crs) if crs is not None else self.crs
        self.data.gst.to_qgis3d(outfile, crs=crs, **kwargs)

    def to_kingdom(
        self,
        outfile: str | Path,
        tdstart: int = 1,
        vw: float = 1500.0,
        vs: float = 1600.0,
    ):
        """
        Write 2 csv files for visualisation of data in Kingdom seismic interpretation
        software:
        - interval data: contains the layer boundaries and properties of the layers.
        - time-depth chart: contains the depth and corresponding time values for the layer
        boundaries.

        Parameters
        ----------
        outfile : str | Path
            Path to csv file to be written.
        tdstart : int
            Startindex for TDchart, default is 1
        vw : float
            Sound velocity in water in m/s, default is 1500 m/s
        vs : float
            Sound velocity in sediment in m/s, default is 1600 m/s

        """
        self.data.gst.to_kingdom(outfile, tdstart, vw, vs)
