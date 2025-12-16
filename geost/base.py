import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, List

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely import geometry as gmt

from geost import config, utils
from geost._warnings import AlignmentWarning
from geost.abstract_classes import AbstractCollection
from geost.projections import (
    horizontal_reference_transformer,
    vertical_reference_transformer,
)
from geost.validation import safe_validate, schemas

type Coordinate = int | float
type GeometryType = gmt.base.BaseGeometry | list[gmt.base.BaseGeometry]
type HeaderObject = Any
type DataObject = Any


class Collection(AbstractCollection):
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
        header: gpd.GeoDataFrame = None,
        data: pd.DataFrame = None,
        has_inclined: bool = False,
        vertical_reference: str | int | CRS = 5709,
    ):
        self._has_inclined = has_inclined
        self._vertical_reference = CRS(vertical_reference)
        self.header = header
        self.data = data

    def __repr__(self):
        if "_header" in self.__dict__:
            return f"{self.__class__.__name__}:\n# header = {self.n_points}"
        else:
            return f"{self.__class__.__name__}:\n<EMPTY COLLECTION>"

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
        return self.header.crs

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

    @header.setter
    def header(self, header):
        if header is not None:
            if config.validation.SKIP:
                self._header = header
            else:
                self._header = safe_validate(
                    header, schemas.pointheader
                )  # TODO: validation schema needs to be inferred
        self.check_header_to_data_alignment()

    @data.setter
    def data(self, data):
        if data is not None:
            if config.validation.SKIP:
                self._data = data
            else:
                self._data = safe_validate(
                    data, schemas.layerdata
                )  # TODO: validation schema needs to be inferred
        self.check_header_to_data_alignment()

    def _clone_with_attrs(self, new_header, new_data):
        """
        Create a deep copy of the current object with new header and data attributes.
        This is used to return a new instance of the collection in methods that modify
        the number of collection objects through e.g. selection and slicing methods.
        Using this method over the self.__class__ constructor ensures that the new
        object only has the header and data attributes updated, while keeping all other
        attributes intact.

        Parameters
        ----------
        new_header : Any Header object
            The new header to be assigned to the cloned collection.
        new_data : Any Data object
            The new data to be assigned to the cloned collection.

        Returns
        -------
        new_collection : New instance of self
            A deep copy of the current object with updated header and data attributes.
        """
        new_collection = deepcopy(self)
        new_collection._header, new_collection._data = new_header, new_data
        return new_collection

    def add_header_column_to_data(self, column_name: str):  # No change
        """
        Add a column from the header to the data table. Useful if you e.g. add some data
        to the header table, but would like to add this to each layer (row in the data
        table) as well.

        Parameters
        ----------
        column_name : str
            Name of the column in the header table to add.

        """
        self._data = pd.merge(self.data, self.header[["nr", column_name]], on="nr")

    def get(self, selection_values: str | Iterable, column: str = "nr"):
        """
        Get a subset of a collection through a string or iterable of object id(s).
        Optionally uses a different column than "nr" (the column with object ids).

        Parameters
        ----------
        selection_values : str | Iterable
            Values to select
        column : str, optional
            In which column of the header to look for selection values, by default "nr"

        Returns
        -------
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        Examples
        --------
        self.get(["obj1", "obj2"]) will return a collection with only these objects.

        Suppose we have a collection of boreholes that we have joined with geological
        map units using the method
        :meth:`~geost.base.PointDataCollection.get_area_labels`. We have added this data
        to the header table in the column 'geological_unit'. Using:

        self.get(["unit1", "unit2"], column="geological_unit")

        will return a :class:`~geost.base.BoreholeCollection` with all boreholes
        that are located in "unit1" and "unit2" geological map areas.
        """
        selected_header = self.header.gsthd.get(selection_values, column)
        selected_data = self.data.gstda.select_by_values(column, selection_values)

        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
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

    def select_within_bbox(
        self,
        xmin: Coordinate,
        ymin: Coordinate,
        xmax: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):
        """
        Make a selection of the collection based on a bounding box.

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
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        """
        selected_header = self.header.gsthd.select_within_bbox(
            xmin, ymin, xmax, ymax, invert=invert
        )
        selected_data = self.data.gstda.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def select_with_points(
        self,
        points: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the collection based on point geometries.

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
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        """
        selected_header = self.header.gsthd.select_with_points(
            points, buffer, invert=invert
        )
        selected_data = self.data.gstda.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def select_with_lines(
        self,
        lines: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the collection based on line geometries.

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
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        """
        selected_header = self.header.gsthd.select_with_lines(
            lines, buffer, invert=invert
        )
        selected_data = self.data.gstda.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def select_within_polygons(
        self,
        polygons: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int = 0,
        invert: bool = False,
    ):
        """
        Make a selection of the collection based on polygon geometries.

        Parameters
        ----------
        polygons : str | Path | gpd.GeoDataFrame | PolygonGeometries
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
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        """
        selected_header = self.header.gsthd.select_within_polygons(
            polygons, buffer=buffer, invert=invert
        )
        selected_data = self.data.gstda.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
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
        selected_header = self.header.gsthd.select_by_depth(
            top_min=top_min, top_max=top_max, end_min=end_min, end_max=end_max
        )
        selected_data = self.data.gstda.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
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
        selected_header = self.header.gsthd.select_by_length(
            min_length=min_length, max_length=max_length
        )
        selected_data = self.data.gstda.select_by_values(
            "nr", selected_header["nr"].unique()
        )
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def select_by_values(
        self, column: str, selection_values: str | Iterable, how: str = "or"
    ):
        """
        Select data based on the presence of given values in a given column. Can be used
        for example to select boreholes that contain peat in the lithology column.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        selection_values : str | Iterable
            Value or values to look for in the column.
        how : str, optional
            Either "and" or "or". "and" requires all selection values to be present in
            column for selection. "or" will select the core if any one of the
            selection_values are found in the column. Default is "and".

        Returns
        -------
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        Examples
        --------
        To select boreholes where both clay ("K") and peat ("V") are present at the same
        time, use "and" as a selection method:

        >>> boreholes.select_by_values("lith", ["V", "K"], how="and")

        To select boreholes that can have one, or both lithologies, use or as the selection
        method:

        >>> boreholes.select_by_values("lith", ["V", "K"], how="and")

        """
        selected_data = self.data.gstda.select_by_values(
            column, selection_values, how=how
        )
        selected_header = self.header.gsthd.get(selected_data["nr"].unique())
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

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
            update_layer_boundaries=update_layer_boundaries,
        )
        selected_header = self.header.gsthd.get(selected_data["nr"].unique())
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def slice_by_values(
        self, column: str, selection_values: str | Iterable, invert: bool = False
    ):
        """
        Slice rows from data based on matching condition. E.g. only return rows with
        a certain lithology in the collection object.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        selection_values : str | Iterable
            Values to look for in the column.
        invert : bool, optional
            If True, invert the slicing action, so remove layers with selected values
            instead of keeping them. The default is False.

        Returns
        -------
        New instance of the current object.
            New instance of the current object containing only the selection resulting
            from application of this method. e.g. if you are calling this method from a
            Collection, you will get an instance of a Collection back.

        Examples
        --------
        Return only rows in borehole data contain sand ("Z") as lithology:

        >>> boreholes.slice_by_values("lith", "Z")

        If you want all the rows that may contain everything but sand, use the "invert"
        option:

        >>> boreholes.slice_by_values("lith", "Z", invert=True)

        """
        selected_data = self.data.gstda.slice_by_values(
            column, selection_values, invert=invert
        )
        selected_header = self.header.gsthd.get(selected_data["nr"].unique())
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

    def select_by_condition(self, condition: Any, invert: bool = False):
        """
        Select from collection.data using a manual condition that results in a boolean
        mask. Returns the rows in the data where the 'condition' evaluates to True.

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
        :class:`~geost.base.LayeredData`
            New instance containing only the data objects selected by this method.

        Examples
        --------
        Select rows in data that contain a specific value:

        >>> data.select_by_condition(data["lith"]=="V")

        Select rows in the data that contain a specific (part of) string or strings:

        >>> boreholes.select_by_condition(boreholes["column"].str.contains("foo|bar"))

        Select rows in data where column values are larger than:

        >>> data.select_by_condition(data["column"] > 2)

        Or select rows in the data based on multiple conditions:

        >>> data.select_by_condition((data["column1"] > 2) & (data["column2] < 1))

        """
        selected_data = self.data.gstda.select_by_condition(condition, invert)
        selected_header = self.header.gsthd.get(selected_data["nr"].unique())
        return self.__class__(
            selected_header, selected_data, self.has_inclined, self.vertical_reference
        )

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
            a column containing the generated data will be added inplace to the header.
        """
        result = self.header.gsthd.get_area_labels(
            polygon_gdf, column_name, include_in_header=include_in_header
        )
        return result

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
        utils._to_geopackage(
            self.header,
            outfile,
            "Header",
            layer="header",
            driver="GPKG",
            index=False,
            metadata=metadata,
        )
        utils._to_geopackage(
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
        >>> collection.to_parquet("example.csv")

        Export the header table:
        >>> collection.to_parquet("example.csv", data_table=False)

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
        utils.save_pickle(self, outfile, **kwargs)

    def to_pyvista_cylinders(
        self,
        displayed_variables: str | List[str],
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
        displayed_variables : str | List[str]
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
        return self.data.to_pyvista_cylinders(
            displayed_variables, radius, vertical_factor, relative_to_vertical_reference
        )

    def to_pyvista_grid(
        self,
        displayed_variables: str | List[str],
        radius: float = 1.0,
    ):
        """
        Create a Pyvista UnstructuredGrid object to represent boreholes. This is more efficient
        than :meth:`~geost.base.Collection.to_pyvista_cylinders` for large datasets, but
        less visually appealing.

        Parameters
        ----------
        displayed_variables : str | List[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m in the MultiBlock. The default is 1.

        Returns
        -------
        pyvista.UniformGrid
            A grid class holding the data which can be iterated over.

        """
        return self.data.to_pyvista_grid(displayed_variables, radius)

    def to_datafusiontools(
        self,
        columns: List[str],
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
        columns : List[str]
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
        List[Data]
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

    def get_cumulative_thickness(
        self, column: str, values: str | List[str], include_in_header: bool = False
    ):
        """
        Get the cumulative thickness of layers where a column contains a specified search
        value or values.

        Parameters
        ----------
        column : str
            Name of column that must contain the search value or values.
        values : str | List[str]
            Search value or values in the column to find the cumulative thickness for.
        include_in_header : bool, optional
            If True, include the result in the header table of the collection. In this
            case, the method does not return a DataFrame. The default is False.

        Returns
        -------
        pd.DataFrame
            Borehole ids and cumulative thickness of selected layers if the "include_in_header"
            option is set to False.

        Examples
        --------
        Get the cumulative thickness of the layers with lithology "K" in the column "lith"
        use:

        >>> boreholes.get_cumulative_thickness("lith", "K")

        Or get the cumulative thickness for multiple selection values. In this case, a
        Pandas DataFrame is returned with a column per selection value containing the
        cumulative thicknesses:

        >>> boreholes.get_cumulative_thickness("lith", ["K", "Z"])

        To include the result in the header object of the collection, use the
        "include_in_header" option:

        >>> boreholes.get_cumulative_thickness("lith", ["K"], include_in_header=True)

        """
        cum_thickness = self.data.gstda.get_cumulative_thickness(column, values)
        cum_thickness.columns = cum_thickness.columns.astype(str)

        if include_in_header:
            columns = [c + "_thickness" for c in cum_thickness.columns]
            self.header.drop(
                columns=columns,
                errors="ignore",
                inplace=True,
            )
            self.header = self.header.merge(
                cum_thickness.add_suffix("_thickness"), on="nr", how="left"
            )
            self.header[columns] = self.header[columns].fillna(0)
        else:
            return cum_thickness

    def get_layer_top(
        self,
        column: str,
        values: str | List[str],
        min_thickness: float = 0,
        min_depth: float = 0,
        include_in_header: bool = False,
    ):
        """
        Find the depth at which a specified layer first occurs, starting at min_depth
        and looking downwards until the first layer of min_thickness is found of the
        specified layer.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        values : str | List[str]
            Value or values of entries in the column that you want to find top of.
        min_thickness : float, optional
            Minimal thickness of the layer to be considered. The default is 0.
        min_depth : float, optional
            Minimal depth of the layer to be considered. The default is 0.
        include_in_header : bool, optional
            If True, include the result in the header table of the collection. In this
            case, the method does not return a DataFrame. The default is False.

        Returns
        -------
        pd.DataFrame
            Borehole ids and top levels of selected layers in meters below the surface.

        Examples
        --------
        Get the top depth of layers in boreholes where the lithology in the "lith" column
        is sand ("Z"):

        >>> boreholes.get_layer_top("lith", "Z")

        To include the result in the header object of the collection, use the
        "include_in_header" option:

        >>> boreholes.get_layer_top("lith", "Z", include_in_header=True)

        """
        tops = self.data.gstda.get_layer_top(
            column, values, min_thickness=min_thickness, min_depth=min_depth
        )

        if include_in_header:
            self.header.drop(
                columns=[c + "_top" for c in tops.columns],
                errors="ignore",
                inplace=True,
            )
            self.header = self.header.merge(
                tops.add_suffix("_top"), on="nr", how="left"
            )
        else:
            return tops

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


class LogCollection(Collection):  # pragma: no cover
    def get_cumulative_thickness(self):
        raise NotImplementedError()

    def get_layer_top(self):
        raise NotImplementedError()
