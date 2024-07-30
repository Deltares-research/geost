import warnings
from pathlib import WindowsPath
from typing import Iterable, List

import geopandas as gpd
import pandas as pd

from geost import spatial
from geost.abstract_classes import AbstractCollection, AbstractData, AbstractHeader
from geost.analysis import cumulative_thickness
from geost.mixins import GeopandasExportMixin, PandasExportMixin
from geost.utils import dataframe_to_geodataframe
from geost.validate.decorators import validate_data, validate_header

type DataObject = DiscreteData | LayeredData
type HeaderObject = LineHeader | PointHeader

type Coordinate = int | float

pd.set_option("mode.copy_on_write", True)


class PointHeader(AbstractHeader, GeopandasExportMixin):
    def __init__(self, gdf):
        self.gdf = gdf
        self.__horizontal_reference = self.gdf.crs

    def __repr__(self):
        return f"{self.__class__.__name__} instance containing {len(self)} objects"

    def __getitem__(self, column):
        return self.gdf[column]

    def __setitem__(self, column, key):
        self.gdf[key] = column

    def __len__(self):
        return len(self.gdf)

    @property
    def gdf(self):
        return self._gdf

    @property
    def horizontal_reference(self):
        return self.__horizontal_reference

    @gdf.setter
    @validate_header
    def gdf(self, gdf):
        self._gdf = gdf

    def get(self, selection_values: str | Iterable, column: str = "nr"):
        """
        Get a subset of a collection through a string or iterable of object id(s).
        Optionally uses a different column than "nr" (the column with object ids).

        Parameters
        ----------
        selection_values : str | Iterable
            Values to select.
        column : str, optional
            In which column of the header to look for selection values, by default "nr".

        Returns
        -------
        Instance of :class:`~geost.headers.PointHeader`.
            Instance of :class:`~geost.headers.PointHeader` containing only
            objects selected through this method.

        Examples
        --------
        self.get(["obj1", "obj2"]) will return a collection with only these objects.

        Suppose we have a collection of boreholes that we have joined with geological
        map units using the method
        :meth:`~geost.headers.PointHeader.get_area_labels`. We have added this data
        to the header table in the column 'geological_unit'. Using:

        self.get(["unit1", "unit2"], column="geological_unit")

        will return a :class:`~geost.headers.PointHeader` with all boreholes
        that are located in "unit1" and "unit2" geological map areas.
        """
        if isinstance(selection_values, str):
            selected_gdf = self[self[column] == selection_values]
        elif isinstance(selection_values, Iterable):
            selected_gdf = self[self[column].isin(selection_values)]

        selected_gdf = selected_gdf[~selected_gdf.duplicated()]
        # selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(selected_gdf)

    def select_within_bbox(
        self,
        xmin: Coordinate,
        xmax: Coordinate,
        ymin: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on a bounding box.

        Parameters
        ----------
        xmin : float | int
            Minimum x-coordinate of the bounding box.
        xmax : float | int
            Maximum x-coordinate of the bounding box.
        ymin : float | int
            Minimum y-coordinate of the bounding box.
        ymax : float | int
            Maximum y-coordinate of the bounding box.
        invert : bool, optional
            Invert the selection, so select all objects outside of the
            bounding box in this case, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.select_points_within_bbox(
            self.gdf, xmin, xmax, ymin, ymax, invert=invert
        )
        return self.__class__(gdf_selected)

    def select_with_points(
        self,
        points: str | WindowsPath | gpd.GeoDataFrame,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on point geometries.

        Parameters
        ----------
        points : str | WindowsPath | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.
        buffer : float | int
            Buffer distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.select_points_near_points(
            self.gdf, points, buffer, invert=invert
        )
        return self.__class__(gdf_selected)

    def select_with_lines(
        self,
        lines: str | WindowsPath | gpd.GeoDataFrame,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on line geometries.

        Parameters
        ----------
        lines : str | WindowsPath | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.
        buffer : float | int
            Buffer distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.select_points_near_lines(
            self.gdf, lines, buffer, invert=invert
        )
        return self.__class__(gdf_selected)

    def select_within_polygons(
        self,
        polygons: str | WindowsPath | gpd.GeoDataFrame,
        buffer: float | int = 0,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on polygon geometries.

        Parameters
        ----------
        polygons : str | WindowsPath | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.
        buffer : float | int, optional
            Optional buffer distance around the polygon selection geometries, by default
            0.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.select_points_within_polygons(
            self.gdf, polygons, buffer, invert=invert
        )
        return self.__class__(gdf_selected)

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
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        selected = self.gdf.copy()
        if top_min is not None:
            selected = selected[selected["mv"] >= top_min]
        if top_max is not None:
            selected = selected[selected["mv"] <= top_max]
        if end_min is not None:
            selected = selected[selected["end"] >= end_min]
        if end_max is not None:
            selected = selected[selected["end"] <= end_max]

        selected = selected[~selected.duplicated()]

        return self.__class__(selected)

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
        Child of :class:`~geost.base.PointDataCollection`.
            Instance of either :class:`~geost.borehole.BoreholeCollection` or
            :class:`~geost.borehole.CptCollection` containing only objects selected by
            this method.
        """
        selected = self.gdf.copy()
        length = selected["mv"] - selected["end"]
        if min_length is not None:
            selected = selected[length >= min_length]
        if max_length is not None:
            selected = selected[length <= max_length]

        selected = selected[~selected.duplicated()]

        return self.__class__(selected)

    def get_area_labels(
        self, polygon_gdf: gpd.GeoDataFrame, column_name: str, include_in_header=False
    ) -> pd.DataFrame:
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
        polygon_gdf = spatial.check_and_coerce_crs(
            polygon_gdf, self.horizontal_reference
        )

        all_nrs = self["nr"]
        area_labels = spatial.find_area_labels(self.gdf, polygon_gdf, column_name)
        area_labels = pd.concat([all_nrs, area_labels], axis=1)

        if include_in_header:
            self._gdf = self.gdf.merge(area_labels, on="nr")
        else:
            return area_labels

    def to_shape(self):
        raise NotImplementedError("Add function logic")

    def to_geoparquet(self):
        raise NotImplementedError("Add function logic")


class LineHeader(AbstractHeader, GeopandasExportMixin):
    def __init__(self, gdf):
        self.gdf = gdf
        self.__horizontal_reference = self.gdf.crs

    def __repr__(self):
        return f"{self.__class__.__name__} instance containing {len(self)} objects"

    def __getitem__(self, column):
        return self.gdf[column]

    def __setitem__(self, column, key):
        self.gdf[key] = column

    def __len__(self):
        return len(self.gdf)

    @property
    def gdf(self):
        return self._gdf

    @property
    def horizontal_reference(self):
        return self.__horizontal_reference

    @gdf.setter
    @validate_header
    def gdf(self, gdf):
        self._gdf = gdf

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


class LayeredData(AbstractData, PandasExportMixin):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __repr__(self):
        name = self.__class__.__name__
        data = self._df
        return f"{name} instance:\n{data}"

    def __getitem__(self, column):
        return self.df[column]

    def __setitem__(self, column, item):
        self.df[column] = item

    def __len__(self):
        return len(self.df)

    @property
    def df(self):
        return self._df

    @df.setter
    # @validate_data
    def df(self, df):
        self._df = df

    def to_header(self):
        header_columns = ["nr", "x", "y", "mv", "end"]
        header = self[header_columns].drop_duplicates().reset_index(drop=True)
        warnings.warn(
            (
                "Header does not contain a crs. Consider setting crs using "
                "header.set_horizontal_reference()."
            )
        )
        header = dataframe_to_geodataframe(header)
        return PointHeader(header)

    def to_collection(self):
        header = self.to_header()
        return BoreholeCollection(
            header, self, 28992, "NAP"
        )  # NOTE: Type of Collection may need to be inferred in the future.

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
        how : str
            Either "and" or "or". "and" requires all selection values to be present in
            column for selection. "or" will select the core if any one of the
            selection_values are found in the column. Default is "and".

        Returns
        -------
        Child of :class:`~geost.base.LayeredData`.
            New instance containing only the data objects selected by this method.

        Examples
        --------
        To select boreholes where both clay ("K") and peat ("V") are present at the same
        time, use "and" as a selection method:

        >>> boreholes.select_by_values("lith", ["V", "K"], how="and")

        To select boreholes that can have one, or both lithologies, use or as the selection
        method:

        >>> boreholes.select_by_values("lith", ["V", "K"], how="and")

        """
        if column not in self.df.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        if isinstance(selection_values, str):
            selection_values = [selection_values]

        selected = self.df.copy()
        if how == "or":
            valid = self["nr"][self[column].isin(selection_values)].unique()
            selected = selected[selected["nr"].isin(valid)]

        elif how == "and":
            for value in selection_values:
                valid = self["nr"][self[column] == value].unique()
                selected = selected[selected["nr"].isin(valid)]

        return self.__class__(selected)

    def slice_depth_interval(self):
        raise NotImplementedError()

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
        invert : bool
            Invert the slicing action, so remove layers with selected values instead of
            keeping them.

        Returns
        -------
        Child of :class:`~geost.base.LayeredData`.
            New instance containing only the data objects selected by this method.

        Examples
        --------
        Return only rows in borehole data contain sand ("Z") as lithology:

        >>> boreholes.slice_by_values("lith", "Z")

        If you want all the rows that may contain everything but sand, use the "invert"
        option:

        >>> boreholes.slice_by_values("lith", "Z", invert=True)

        """
        if isinstance(selection_values, str):
            selection_values = [selection_values]

        sliced = self.df.copy()

        if invert:
            sliced = sliced[~sliced[column].isin(selection_values)]
        else:
            sliced = sliced[sliced[column].isin(selection_values)]

        return self.__class__(sliced)

    def get_cumulative_layer_thickness(self, column: str, values: str | List[str]):
        """
        Get the cumulative thickness of layers where a column contains a specified search
        value or values.

        Parameters
        ----------
        column : str
            Name of column that must contain the search value or values.
        values : str | List[str]
            Search value or values in the column to find the cumulative thickness for.

        Returns
        -------
        pd.DataFrame
            Borehole ids and cumulative thickness of selected layers.

        Examples
        --------
        Get the cumulative thickness of the layers with lithology "K" in the column "lith"
        use:

        >>> boreholes.get_cumulative_layer_thickness("lith", "K")

        Or get the cumulative thickness for multiple selection values. In this case, a
        Pandas DataFrame is returned with a column per selection value containing the
        cumulative thicknesses:

        >>> boreholes.get_cumulative_layer_thickness("lith", ["K", "Z"])

        """
        selected_layers = self.slice_by_values(column, values)

        cum_thickness = selected_layers.df.groupby(["nr", column]).apply(
            cumulative_thickness
        )
        return cum_thickness.unstack(level=column)

    def get_layer_top(self, column: str, values: str | List[str]):
        """
        Find the depth at which a specified layer first occurs.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        values : str | List[str]
            Value or values of entries in the column that you want to find top of.

        Returns
        -------
        pd.DataFrame
            Borehole ids and top levels of selected layers in meters below the surface.

        Examples
        --------
        Get the top depth of layers in boreholes where the lithology in the "lith" column
        is sand ("Z"):

        >>> boreholes.get_layer_top("lith", "Z")

        """
        selected_layers = self.slice_by_values(column, values)
        layer_top = selected_layers.df.groupby(["nr", column])["top"].first()
        return layer_top.unstack(level=column)

    def to_vtm(self):
        raise NotImplementedError()

    def to_datafusiontools(self):
        raise NotImplementedError()


class DiscreteData(AbstractData, PandasExportMixin):
    def __init__(self, df):
        raise NotImplementedError(f"{self.__class__.__name__} not supported yet")

    @property
    def df(self):
        return self._df

    @df.setter
    @validate_data
    def df(self, df):
        self._df = df

    def to_header(self):
        raise NotImplementedError()

    def to_collection(self):
        raise NotImplementedError()

    def select_by_values(self):
        raise NotImplementedError()

    def slice_depth_interval(self):
        raise NotImplementedError()

    def slice_by_values(self):
        raise NotImplementedError()

    def get_cumulative_layer_thickness(self):
        raise NotImplementedError()
        pass

    def get_layer_top(self):
        raise NotImplementedError()

    def to_vtm(self):
        raise NotImplementedError()

    def to_datafusiontools(self):
        raise NotImplementedError()


class Collection(AbstractCollection):
    def __init__(
        self,
        header: HeaderObject,
        data: DataObject,
        horizontal_reference: int,
        vertical_reference: str,
    ):
        self.horizontal_reference = horizontal_reference
        self.vertical_reference = vertical_reference
        self.header = header
        self.data = data

    def __new__(cls, *args, **kwargs):
        if cls is Collection:
            raise TypeError(
                f"Cannot construct {cls.__name__} directly: construct class from its",
                "children instead",
            )
        else:
            return object.__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n# header = {self.n_points}"

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        return self._data

    @property
    def n_points(self):  # No change
        """
        Number of objects in the collection.
        """
        return len(self.header.df)

    @property
    def horizontal_reference(self):  # Move to header class in future refactor
        return self._horizontal_reference

    @property
    def vertical_reference(self):  # move to data class in future refactor
        return self._vertical_reference

    @header.setter
    def header(self, header):
        if isinstance(header, LineHeader | PointHeader):
            self._header = header
        elif "_header" in self.__dict__.keys() and isinstance(header, gpd.GeoDataFrame):
            self._header = self._header.__class__(header)
        self.check_header_to_data_alignment()

    @data.setter
    def data(self, data):
        if isinstance(data, LayeredData | DiscreteData):
            self._data = data
        elif "_data" in self.__dict__.keys() and isinstance(data, pd.DataFrame):
            self._data = self._header.__class__(data)
        self.check_header_to_data_alignment()

    def horizontal_reference(self, to_epsg):
        raise NotImplementedError("Add function logic")

    def vertical_reference(self, to_epsg):
        raise NotImplementedError("Add function logic")

    def get(self):
        raise NotImplementedError("Add function logic")

    def reset_header(self):
        raise NotImplementedError("Add function logic")

    def check_header_to_data_alignment(self):
        pass

    def check_and_coerce_crs(self):
        pass

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

    def select_by_values(self):
        raise NotImplementedError("Add function logic")

    def slice_depth_interval(self):
        raise NotImplementedError("Add function logic")

    def slice_by_values(self):
        raise NotImplementedError("Add function logic")

    def get_cumulative_layer_thickness(self):
        # Not sure if this should be here, potentially unsuitable with DiscreteData
        raise NotImplementedError("Add function logic")

    def get_layer_top(self):
        raise NotImplementedError("Add function logic")

    def to_vtm(self):
        raise NotImplementedError("Add function logic")

    def to_datafusiontools(self):
        # supporting this is low priority, perhaps even deprecate
        raise NotImplementedError("Add function logic")


class BoreholeCollection(Collection):
    pass


class CptCollection(Collection):
    pass


class LogCollection(Collection):
    pass
