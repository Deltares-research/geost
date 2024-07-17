import pickle
from functools import reduce
from pathlib import WindowsPath
from typing import Iterable, List, Optional, TypeVar

import pandas as pd

# Local imports
from geost import spatial
from geost.analysis import cumulative_thickness, layer_top
from geost.export import borehole_to_multiblock, export_to_dftgeodata
from geost.projections import get_transformer
from geost.utils import ARITHMIC_OPERATORS, inform_user, warn_user
from geost.validate.decorators import validate_data, validate_header

warn = warn_user(lambda warning_info: print(warning_info))
inform = inform_user(lambda info: print(info))

Coordinate = TypeVar("Coordinate", int, float)
GeoDataFrame = TypeVar("GeoDataFrame")
DataArray = TypeVar("DataArray")

pd.set_option("mode.copy_on_write", True)


class PointDataCollection:
    """
    Base class for collections of pointdata, such as boreholes and CPTs. The geost
    module revolves around this class and includes all methods that apply generically to
    all types of point data, such as selection and export methods.

    This class cannot be constructed directly, but only from
    :class:`~geost.borehole.BoreholeCollection` and
    :class:`~geost.borehole.CptCollection`. Users must use the reader functions in
    :py:mod:`~geost.read` to create collections.

    Args:
        data (pd.DataFrame): Dataframe containing borehole/CPT data.

        vertical_reference (str): Vertical reference, see
         :py:attr:`~geost.base.PointDataCollection.vertical_reference`

        horizontal_reference (int): Horizontal reference, see
         :py:attr:`~geost.base.PointDataCollection.horizontal_reference`

        header (pd.DataFrame): Header used for construction. see
         :py:attr:`~geost.base.PointDataCollection.header`
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vertical_reference: str,
        horizontal_reference: int,
        header: Optional[pd.DataFrame] = None,
        is_inclined: bool = False,
    ):
        self.__vertical_reference = vertical_reference
        self.__horizontal_reference = horizontal_reference
        self.__is_inclined = is_inclined
        self.data = data
        if isinstance(header, pd.DataFrame):
            self.header = header
        else:
            self.reset_header()

    def __new__(cls, *args, **kwargs):
        if cls is PointDataCollection:
            raise TypeError(
                f"Cannot construct {cls.__name__} directly: construct class from its",
                "children instead",
            )
        else:
            return object.__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n# header = {self.n_points}"

    def get(self, selection_values: str | Iterable, column: str = "nr"):  # Header class
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
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.

        Examples
        --------
        self.get(["obj1", "obj2"]) will return a collection with only these objects.

        Suppose we have a collection of boreholes that we have joined with geological
        map units using the method
        :meth:`~geost.base.PointDataCollection.get_area_labels`. We have added this data
        to the header table in the column 'geological_unit'. Using:

        self.get(["unit1", "unit2"], column="geological_unit")

        will return a :class:`~geost.borehole.BoreholeCollection` with all boreholes
        that are located in "unit1" and "unit2" geological map areas.
        """
        if isinstance(selection_values, str):
            selected_header = self.header[self.header[column] == selection_values]
        elif isinstance(selection_values, Iterable):
            selected_header = self.header[self.header[column].isin(selection_values)]

        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def reset_header(
        self, *args
    ):  # Stays the same here, calls 'to_collection' from Data class
        """
        Create a new header based on the 'data' dataframe
        (:py:attr:`~geost.base.PointDataCollection.data`). Can be used to reset the
        header in case you accidentally broke the header.

        Raises
        ------
        TypeError
            If len(header_col_names) != 5
        """
        header_col_names = ["nr", "x", "y", "mv", "end"]
        header = self.data.drop_duplicates(subset=header_col_names[0])
        header = header[header_col_names].reset_index(drop=True)
        self.header = spatial.dataframe_to_geodataframe(
            header, self.horizontal_reference
        )

    @property
    def header(self):  # Calls property like 'self.header.df'
        """
        Pandas dataframe of header (1 row per object in the collection) and includes at
        the minimum: point id, x-coordinate, y-coordinate, surface level and end depth:

        Column names:
        ["nr", "x", "y", "mv", "end"]

        Extra data can be added through various methods or manually by the user.
        However, the above columns must always be present. Every time the header is
        changed in some way, it runs through validation to warn the user of any
        potential problems
        """
        return self._header

    @property
    def data(self):  # Calls property like 'self.data.df'
        """
        Pandas dataframe that contains all data of objects in the collection. e.g. all
        layers of a borehole.

        Extra data can be added through various methods or manually by the user.
        Every time the data attribute is changed in some way, it runs through validation
        to warn the user of any potential problems.
        """
        return self._data

    @property
    def n_points(self):  # No change
        """
        Number of objects in the collection.
        """
        return len(self.header)

    @property
    def vertical_reference(self):  # Discuss for refactor Positional reference
        """
        Current vertical reference system of the collection.

        Returns
        -------
        str
            Vertical reference. Either 'NAP', 'surfacelevel' or 'depth'

            'NAP' is the elevation with respect to NAP datum.

            'surfacelevel' is elevation with respect to surface (surface is 0 m, e.g.
            layers tops could be 0, -1, -2 etc.).

            'depth' is depth with respect to surface (surface is 0 m, e.g. depth of
            layers tops could be 0, 1, 2 etc.).
        """
        return self.__vertical_reference

    @property
    def horizontal_reference(self):  # Discuss for refactor Positional reference
        """
        Current horizontal reference of the collection. The horizontal reference must
        be correct in order for spatial selection functions to work.

        Returns
        -------
        int
            EPSG code of the coordinate reference system (crs) used for point geometries
            in :py:attr:`~geost.base.PointDataCollection.header`
        """
        return self.__horizontal_reference

    @property
    def is_inclined(self):  # Data class property
        """
        Whether borehole/cpt/log is inclined.

        Returns
        -------
        bool
            True if one or more of the objects in the collection are inclined. False
            if all objects go straight downward.
        """
        return self.__is_inclined

    @header.setter
    @validate_header
    def header(self, header):  # No change
        """
        This setter is called whenever the attribute 'header' is manipulated, either
        during construction (init) or when a user attempts to set the header on an
        instance of the object (e.g. instance.header = new_header_df). This will run the
        header through validation and warn the user of any potential problems.
        """
        self._header = header
        self.__check_header_to_data_alignment()

    @data.setter
    @validate_data
    def data(self, data):  # No change
        """
        This setter is called whenever the attribute 'data' is manipulated, either
        during construction (init) or when a user attempts to set the data on an
        instance of the object (e.g. instance.data = new_data_df). This will run the
        data through validation and warn the user of any potential problems.
        """
        self._data = data
        self.__check_header_to_data_alignment()

    def __check_header_to_data_alignment(self):  # No change
        """
        Two-way check to warn of any misalignment between the header and data
        attributes. Two way, i.e. if header includes more objects than in the data and
        if the data includes more unique objects that listed in the header.

        This check is performed everytime the object is instantiated AND if any change
        is made to either the header or data attributes (see their respective setters).
        """
        if hasattr(self, "_header") and hasattr(self, "_data"):
            if any(~self.header["nr"].isin(self.data["nr"].unique())):
                warn(
                    "Header covers more objects than present in the data table, "
                    "consider running the method 'reset_header' to update the header."
                )
            if not set(self.data["nr"].unique()).issubset(set(self.header["nr"])):
                warn(
                    "Header does not cover all unique objects in data, consider "
                    "running the method 'reset_header' to update the header."
                )

    def __check_and_coerce_crs(
        self, other_gdf: GeoDataFrame
    ) -> GeoDataFrame:  # No change untill positional reference refactor
        """
        Check the CRS of a geodataframe against the collection's current horizontal
        reference :py:attr:`~geost.base.PointDataCollection.horizontal_reference`. This
        method is called whenever actions that compare the PointdataCollection
        geometries with other geometries are performed.

        If the other dataframe has no CRS, warn user and assume CRS is the same as
        :py:attr:`~geost.base.PointDataCollection.horizontal_reference`.

        If the other dataframe has a different known CRS, inform user and coerce the CRS
        to :py:attr:`~geost.base.PointDataCollection.horizontal_reference`.

        Parameters
        ----------
        other_gdf : GeoDataFrame
            Other geodataframe to check for having the same CRS as the
            PointdataCollection.

        Returns
        -------
        GeoDataFrame
            Other geodataframe coerced to have the same CRS as the PointdataCollection.
        """
        if other_gdf.crs is None:
            other_gdf.crs = self.horizontal_reference
            warn(
                "The selection geometry has no crs! Assuming it is the same as the "
                + f"horizontal_reference (epsg:{self.horizontal_reference}) of this "
                + "collection. PLEASE CHECK WHETHER THIS IS CORRECT!",
            )
        elif other_gdf.crs != self.horizontal_reference:
            other_gdf = other_gdf.to_crs(self.horizontal_reference)
            inform(
                "The crs of the selection geometry does not match the horizontal "
                + "reference of the collection. The selection geometry was coerced "
                + f"to epsg:{self.horizontal_reference} automatically"
            )
        return other_gdf

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
        self.data = pd.merge(self.data, self.header[["nr", column_name]], on="nr")

    def change_vertical_reference(
        self, to: str
    ):  # Discuss for refactor Positional reference
        """
        Change the vertical reference of layer tops and bottoms

        Parameters
        ----------
        to : str
            To which vertical reference to convert the layer tops and bottoms. Either
            'NAP', 'surfacelevel' or 'depth'. See
            :py:attr:`~geost.base.PointDataCollection.vertical_reference`.
        """
        match self.__vertical_reference:
            case "NAP":
                if to == "surfacelevel":
                    self._data["top"] = self._data["top"] - self._data["mv"]
                    self._data["bottom"] = self._data["bottom"] - self._data["mv"]
                    self.__vertical_reference = "surfacelevel"
                elif to == "depth":
                    self._data["top"] = (self._data["top"] - self._data["mv"]) * -1
                    self._data["bottom"] = (
                        self._data["bottom"] - self._data["mv"]
                    ) * -1
                    self.__vertical_reference = "depth"
            case "surfacelevel":
                if to == "NAP":
                    self._data["top"] = self._data["top"] + self._data["mv"]
                    self._data["bottom"] = self._data["bottom"] + self._data["mv"]
                    self.__vertical_reference = "NAP"
                if to == "depth":
                    self._data["top"] = self._data["top"] * -1
                    self._data["bottom"] = self._data["bottom"] * -1
                    self.__vertical_reference = "depth"
            case "depth":
                if to == "NAP":
                    self._data["top"] = self._data["top"] * -1 + self._data["mv"]
                    self._data["bottom"] = self._data["bottom"] * -1 + self._data["mv"]
                    self.__vertical_reference = "NAP"
                if to == "surfacelevel":
                    self._data["top"] = self._data["top"] * -1
                    self._data["bottom"] = self._data["bottom"] * -1
                    self.__vertical_reference = "surfacelevel"

    def change_horizontal_reference(
        self, target_crs: int, only_geometries: bool = True
    ):  # Discuss for refactor Positional reference, probably both Header and Data classes contain this method
        """
        Change the horizontal reference (i.e. coordinate reference system, crs) of the
        collection to the given target crs.

        Parameters
        ----------
        to_crs : int
            EPSG of the target crs
        only_geometries : bool, optional
            Only transform the point geometries, but leave the 'x' and 'y' columns in
            both header and data in the original crs. Only converting geometries is much
            faster and ensures that geometry exports of the collection behave as
            expected, by default True.
        """
        self._header = self._header.to_crs(target_crs)
        if not only_geometries:
            # Create Pyproj transformer from this collection's crs to target crs
            transformer = get_transformer(self.horizontal_reference, target_crs)
            self._header["x"], self._header["y"] = transformer.transform(
                self._header["x"], self._header["y"]
            )
            self._data["x"], self._data["y"] = transformer.transform(
                self._data["x"], self._data["y"]
            )

            if self.is_inclined:
                self._data["x_bot"], self._data["y_bot"] = transformer.transform(
                    self._data["x_bot"], self._data["y_bot"]
                )

        self.__horizontal_reference = target_crs

    def select_within_bbox(
        self,
        xmin: Coordinate,
        xmax: Coordinate,
        ymin: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):  # Header class
        """
        Make a selection of the data based on a bounding box of coordinates in the
        horizontal reference system of the collection. See also
        :py:attr:`~geost.base.PointDataCollection.horizontal_reference`

        Parameters
        ----------
        xmin : Coordinate (float or int).
            Left x-coordinate of bbox.
        xmax : Coordinate (float or int).
            Right x-coordinate of bbox.
        ymin : Coordinate (float or int).
            Lower y-coordinate of bbox.
        ymax : Coordinate (float or int).
            Upper y-coordinate of bbox.
        invert: bool, default False.
            Invert the selection.

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        selected_header = spatial.header_from_bbox(
            self.header, xmin, xmax, ymin, ymax, invert
        )
        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def select_with_points(
        self,
        point_gdf: GeoDataFrame,
        buffer: float = 100,
        invert: bool = False,
    ):  # Header class
        """
        Make a selection of the data based on points.

        Parameters
        ----------
        line_file : str | WindowsPath
            Shapefile or geopackage containing point data.
        buffer: float, default 100
            Buffer around the lines to select points. Default 100.
        invert: bool, default False
            Invert the selection.

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        point_gdf = self.__check_and_coerce_crs(point_gdf)

        selected_header = spatial.header_from_points(
            self.header, point_gdf, buffer, invert
        )
        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def select_with_lines(
        self,
        line_gdf: GeoDataFrame,
        buffer: float = 100,
        invert: bool = False,
    ):  # Header class
        """
        Make a selection of the data based on lines.

        Parameters
        ----------
        line_file : str | WindowsPath
            Shapefile or geopackage containing linestring data.
        buffer: float, default 100
            Buffer around the lines to select points. Default 100.
        invert: bool, default False
            Invert the selection.

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        line_gdf = self.__check_and_coerce_crs(line_gdf)

        selected_header = spatial.header_from_lines(
            self.header, line_gdf, buffer, invert
        )
        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def select_within_polygons(
        self,
        polygon_gdf: GeoDataFrame,
        buffer: float = 0,
        invert: bool = False,
    ):  # Header class
        """
        Make a selection of the data based on polygons.

        Parameters
        ----------
        polygon_file : str | WindowsPath
            Shapefile or geopackage containing (multi)polygon data.
        invert: bool, default False
            Invert the selection.

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        polygon_gdf = self.__check_and_coerce_crs(polygon_gdf)

        selected_header = spatial.header_from_polygons(
            self.header, polygon_gdf, buffer, invert
        )
        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def select_by_values(
        self, column: str, selection_values: str | Iterable, how: str = "or"
    ):  # Data class
        """
        Select pointdata based on the presence of given values in the given columns.
        Can be used for example to return a BoreholeCollection of boreholes that contain
        peat in the lithology column. This can be achieved by passing e.g. the following
        arguments to the method:

        self.select_by_values("lith", ["V", "K"], how="and"):
        Returns boreholes where lithoclasses "V" and "K" are present at the same time.

        self.select_by_values("lith", ["V", "K"], how="or"):
        Returns boreholes where either lithoclasses "V" or "K" are present
        (or both by coincidence)

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        selection_values : str | Iterable
            Values to look for in the column.
        how : str
            Either "and" or "or". "and" requires all selction values to be present in
            column for selection. "or" will select the core if any one of the
            selection_values are found in the column. Default is "and".

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        if column not in self.data.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        if isinstance(selection_values, str):
            selection_values = [selection_values]

        header_copy = self.header.copy()
        if how == "or":
            notna = self.data["nr"][self.data[column].isin(selection_values)].unique()
            selected_header = header_copy[header_copy["nr"].isin(notna)]
        elif how == "and":
            for selection_value in selection_values:
                notna = self.data["nr"][
                    self.data[column].isin([selection_value])
                ].unique()
                header_copy = header_copy[header_copy["nr"].isin(notna)]
            selected_header = header_copy

        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def select_by_depth(
        self,
        top_min: float = None,
        top_max: float = None,
        end_min: float = None,
        end_max: float = None,
        slice: bool = False,
    ):  # Header class
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
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        selected_header = self.header.copy()
        if top_min is not None:
            selected_header = selected_header[selected_header["mv"] >= top_min]
        if top_max is not None:
            selected_header = selected_header[selected_header["mv"] <= top_min]
        if end_min is not None:
            selected_header = selected_header[selected_header["end"] >= end_min]
        if end_max is not None:
            selected_header = selected_header[selected_header["end"] <= end_max]

        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def select_by_length(
        self, min_length: float = None, max_length: float = None
    ):  # Header class
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
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        selected_header = self.header.copy()
        length = selected_header["mv"] - selected_header["end"]
        if min_length is not None:
            selected_header = selected_header[length >= min_length]
        if max_length is not None:
            selected_header = selected_header[length <= max_length]

        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=selected_header,
            is_inclined=self.is_inclined,
        )

    def slice_depth_interval(
        self,
        upper_boundary: float | int = None,
        lower_boundary: float | int = None,
        vertical_reference: Optional[str] = None,
        update_layer_boundaries: bool = True,  # TODO: implement
    ):  # Data class
        """
        Slice boreholes/cpts based on given upper and lower boundaries. This returns an
        instance of the BoreholeCollection or CptCollection containing only the sliced
        layers.

        Note #1: This method currently only slices along existing layer boundaries,
        which especially for boreholes could mean that thick layers may continue beyond
        the given boundaries.

        Note #2: The instance that is returned may contain a smaller number of objects
        if the slicing led to a removal of all layers of an object.

        Parameters
        ----------
        upper_boundary : float | int, optional
            Every layer that starts above this is removed, by default 9999.
        lower_boundary : float | int, optional
            Every layer that starts below this is removed, by default -9999.
        vertical_reference : str, optional
            The vertical reference used in slicing. Either "NAP", "surface" or "depth"
            See documentation of the change_vertical_reference method for details
            on the possible vertical references. By default "NAP".

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing depth-sliced objects
            resulting from applying this method.

        Examples
        --------
        Usage depends on the vertical_reference of the ~geost.base.PointDataCollection. For
        example, if you want to cut off all layers below -10 m NAP and all layers above 2 m
        NAP in a collection that has NAP as its reference, then:

        >>> self.slice_vertical(upper_boundary=2, lower_boundary=-10)

        This uses the vertical_reference of the collection the slice is performed on. If you
        want to do a selection with respect to 'depth' or 'surfacelevel' references use:

        >>> self.slice_vertical(upper_boundary=2, lower_boundary=6, vertical_reference='depth')
        >>> self.slice_vertical(upper_boundary=-2, lower_boundary=-6, vertical_reference='surfacelevel')

        """
        if vertical_reference is None:
            vertical_reference = self.vertical_reference
            change_reference = False
        else:
            change_reference = vertical_reference != self.vertical_reference

        if change_reference:
            original_vertical_reference = self.vertical_reference
            self.change_vertical_reference(vertical_reference)

        data_sliced = self.data.copy()

        if vertical_reference != "depth":
            data_sliced = data_sliced[data_sliced["top"] > (lower_boundary or -1e34)]
            data_sliced = data_sliced[data_sliced["bottom"] < (upper_boundary or 1e34)]
        elif vertical_reference == "depth":
            data_sliced = data_sliced[data_sliced["top"] < (lower_boundary or 1e34)]
            data_sliced = data_sliced[data_sliced["bottom"] > (upper_boundary or 1)]

        header_sliced = self.header.loc[
            self.header["nr"].isin(data_sliced["nr"].unique())
        ]

        result = self.__class__(
            data_sliced,
            vertical_reference=vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=header_sliced,
            is_inclined=self.is_inclined,
        )

        if change_reference:
            self.change_vertical_reference(original_vertical_reference)

        return result

    def slice_by_values(
        self, column: str, selection_values: str | Iterable, invert: bool = False
    ):  # Data class
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
        invert : bool
            Invert the slicing action, so remove layers with selected values instead of
            keeping them.

        Returns
        -------
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing depth-sliced objects
            resulting from applying this method.
        """
        if isinstance(selection_values, str):
            selection_values = [selection_values]

        data_sliced = self.data.copy()
        if invert:
            data_sliced = data_sliced[~data_sliced[column].isin(selection_values)]
        elif not invert:
            data_sliced = data_sliced[data_sliced[column].isin(selection_values)]

        header_sliced = self.header.loc[
            self.header["nr"].isin(data_sliced["nr"].unique())
        ]

        result = self.__class__(
            data_sliced,
            vertical_reference=self.vertical_reference,
            horizontal_reference=self.horizontal_reference,
            header=header_sliced,
            is_inclined=self.is_inclined,
        )

        return result

    def get_area_labels(
        self, polygon_gdf: GeoDataFrame, column_name: str, include_in_header=False
    ) -> pd.DataFrame:  # Header class
        """
        Find in which area (polygons) the point data locations fall. e.g. to determine
        in which geomorphological unit points are located.

        Parameters
        ----------
        polygon_gdf : gpd.GeoDataFrame
            GeoDataFrame with polygons.
        column_name : str
            The column name to find the labels in.
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default
            False.

        Returns
        -------
        pd.DataFrame
            Borehole ids and the polygon label they are in. If include_in_header = True,
            a column containing the generated data will be added inplace to
            :py:attr:`~geost.base.PointDataCollection.header`.
        """
        polygon_gdf = self.__check_and_coerce_crs(polygon_gdf)

        all_nrs = self.header["nr"]
        area_labels = spatial.find_area_labels(self.header, polygon_gdf, column_name)
        area_labels = pd.concat([all_nrs, area_labels], axis=1)

        if include_in_header:
            self._header = self.header.merge(area_labels, on="nr")
        else:
            return area_labels

    def get_cumulative_layer_thickness(
        self, column: str, values: str | List[str], include_in_header=False
    ):  # Data class
        """
        Get the cumulative thickness of layers of a certain type.

        For example, to get the cumulative thickness of the layers with lithology "K" in
        the column "lith" use:

        self.get_cumulative_layer_thickness("lith", "K")

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        values : str | List[str]
            Value(s) of entries in column that you want to find the cumulative thickness
            of.
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default
            False.

        Returns
        -------
        pd.DataFrame
            Borehole ids and cumulative thickness of selected layers. If
            include_in_header = True, a column containing the generated data will be
            added inplace to :py:attr:`~geost.base.PointDataCollection.header`.
        """
        if isinstance(values, str):
            values = [values]

        result_dfs = []
        for value in values:
            cumulative_thicknesses = cumulative_thickness(self.data, column, value)
            result_df = pd.DataFrame(
                cumulative_thicknesses, columns=("nr", f"{value}_thickness")
            )
            result_dfs.append(result_df)

        result = reduce(lambda left, right: pd.merge(left, right, on="nr"), result_dfs)
        if include_in_header:
            self._header = self.header.merge(result, on="nr")
        else:
            return result

    def get_layer_top(
        self, column: str, values: str | List[str], include_in_header=False
    ):  # Data class
        """
        Find the depth at which a specified layer first occurs.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        values : str | List[str]
            Value of entries in column that you want to find top of.
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default
            False.

        Returns
        -------
        pd.DataFrame
            Borehole ids and top levels of selected layers. If
            include_in_header = True, a column containing the generated data will be
            added inplace to :py:attr:`~geost.base.PointDataCollection.header`.
        """
        if isinstance(values, str):
            values = [values]

        result_dfs = []
        for value in values:
            layer_tops = layer_top(self.data, column, value)
            result_df = pd.DataFrame(layer_tops, columns=("nr", f"{value}_top"))
            result_dfs.append(result_df)

        result = reduce(lambda left, right: pd.merge(left, right, on="nr"), result_dfs)
        if include_in_header:
            self._header = self.header.merge(result, on="nr")
        else:
            return result

    def append(self, other):  # No (significant) change
        """
        Append data of other object of the same type (e.g BoreholeCollection to
        BoreholeCollection).

        Parameters
        ----------
        other : Instance of the same type as self.
            Another object of the same type, from which the data is appended to self.

        Raises
        ------
        TypeError
            If the instance 'other' is not of the same type as self (e.g. when
            attempting to append a CptCollection to a BoreholeCollection).
        """
        if (
            self.__class__ == other.__class__
            and self.horizontal_reference == other.horizontal_reference
        ):
            # Check overlap first and remove duplicates from 'other' if required
            other_header_overlap = other.header["nr"].isin(self.header["nr"])
            if any(other_header_overlap):
                other_header = other.header[~other_header_overlap]
                other_data = other.data.loc[other.data["nr"].isin(other_header["nr"])]
            else:
                other_header = other.header
                other_data = other.data

            self._data = pd.concat([self.data, other_data])
            self._header = pd.concat([self.header, other_header])
            # TODO return newly constructed class from header and data combination
            # once class construction from header and data is implemented.
        else:
            raise TypeError(
                f"Cannot join instance of {self.__class__} with an instance of ",
                f"{other.__class__}: collection types AND horizontal references must ",
                "match",
            )

    def to_parquet(self, out_file: str | WindowsPath, **kwargs):  # Data class
        """
        Write data to parquet file.

        Parameters
        ----------
        out_file : str | WindowsPath
            Path to parquet file to be written.
        **kwargs
            pd.DataFrame.to_parquet kwargs. See relevant Pandas documentation.
        """
        self._data.to_parquet(out_file, **kwargs)

    def to_csv(self, out_file: str | WindowsPath, **kwargs):  # Data class
        """
        Write data to csv file.

        Parameters
        ----------
        out_file : str | WindowsPath
            Path to csv file to be written.
        **kwargs
            pd.DataFrame.to_csv kwargs. See relevant Pandas documentation.
        """
        self._data.to_csv(out_file, **kwargs)

    def to_shape(self, out_file: str | WindowsPath, **kwargs):  # Header class
        """
        Write header data to shapefile or geopackage. You can use the resulting file to
        display borehole locations in GIS for instance.

        Parameters
        ----------
        out_file : str | WindowsPath
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_file kwargs. See relevant GeoPandas documentation.
        """
        self.header.to_file(out_file, **kwargs)

    def to_geoparquet(self, out_file: str | WindowsPath, **kwargs):  # Header class
        """
        Write header data to geoparquet. You can use the resulting file to display
        borehole locations in GIS for instance. Please note that Geoparquet is supported
        by GDAL >= 3.5. For Qgis this means QGis >= 3.26

        Parameters
        ----------
        out_file : str | WindowsPath
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_parquet kwargs. See relevant Pandas documentation.
        """
        self.header.to_parquet(out_file, **kwargs)

    def to_ipf(self, out_file: str | WindowsPath, **kwargs):  # Not implemented
        # TODO write the pandas dataframes to IPF
        pass

    def to_vtm(
        self,
        out_file: str | WindowsPath,
        data_columns: List[str],
        radius: float = 1,
        vertical_factor: float = 1.0,
        **kwargs,
    ):  # Data class
        """
        Save objects to VTM (Multiblock file, an XML VTK file pointing to multiple other
        VTK files). For viewing boreholes/cpt's in e.g. ParaView or other VTK viewers.

        Parameters
        ----------
        out_file : str | WindowsPath
            Path to vtm file to be written.
        data_columns : List[str]
            Labels of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m, by default 1.
        vertical_factor : float, optional
            Factor to correct vertical scale. e.g. when layer boundaries are given in cm
            use 0.01 to convert to m, by default 1.0. It is not recommended to use this
            for vertical exaggeration, use viewer functionality for that instead.
        **kwargs :
            pyvista.MultiBlock.save kwargs. See relevant Pyvista documentation.
        """
        if not self.__vertical_reference == "NAP":
            raise NotImplementedError(
                'VTM export is not available for other vertical references than "NAP"'
            )
        if self.is_inclined:
            raise NotImplementedError(
                "VTM export of inclined objects is not yet supported"
            )

        vtk_object = borehole_to_multiblock(
            self.data, data_columns, radius, vertical_factor, **kwargs
        )
        vtk_object.save(out_file, **kwargs)

    def to_datafusiontools(
        self,
        columns: List[str],
        out_file: str | WindowsPath = None,
        encode: bool = False,
        **kwargs,
    ):  # Data class
        """
        Write a collection to the core "Data" class of Deltares DataFusionTools. Returns
        a list of "Data" objects, one for each object in the Borehole/CptCollection that
        you exported. This list can directly be used within DataFusionTools. If out_file
        is given, the list of Data objects is saved to a pickle file.

        Warning: categorical data is optionally encoded (all possible values become
        a seperate feature that is 0 or 1). If there is a large number of possible
        categories the export process may become slow. Please consider carefully which
        categorical data columns need to be included.

        For DataFusionTools visit:
        https://bitbucket.org/DeltaresGEO/datafusiontools/src/master/

        Parameters
        ----------
        columns : List[str]
            Which columns (in the self.data dataframe) to include.
        out_file : str | WindowsPath
            Path to pickle file to be written.
        encode : bool, default True
            Encode categorical data to additional binary columns (0 or 1).
            Also see explanation above. Default is False.
        """
        if not self.__vertical_reference == "NAP":
            raise NotImplementedError(
                "DataFusionTools export is not available for other vertical references"
                ' than "NAP"'
            )

        dftgeodata = export_to_dftgeodata(self.data, columns, encode=encode)

        if out_file:
            with open(out_file, "wb") as f:
                pickle.dump(dftgeodata, f)
        else:
            return dftgeodata
