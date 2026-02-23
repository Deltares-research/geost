from __future__ import annotations

from functools import singledispatchmethod, wraps
from typing import TYPE_CHECKING, Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import buffer, linestrings

import geost
from geost import (
    export,
    spatial,
    validation,
)  # FIXME: spatial triggers import of xarray, rioxarray. We don't want this automatically.
from geost.validation.method_checks import _requires_depth, _requires_geometry

if TYPE_CHECKING:
    from pathlib import Path

    from pandera import DataFrameSchema
    from pyproj import CRS
    from shapely.geometry.base import BaseGeometry

type DataFrame = pd.DataFrame | gpd.GeoDataFrame
type GeometryType = BaseGeometry | list[BaseGeometry]


@pd.api.extensions.register_dataframe_accessor("gst")
class GeostFrame:
    def __init__(self, dataframe: DataFrame):
        self._validate_dataframe(dataframe)
        self._obj = dataframe
        self._set_depth_columns()

    def _set_depth_columns(self):
        """
        Determine which columns in the DataFrame contain depth information and set them
        as attributes of the accessor. The method looks for the presence of either "top"
        and "bottom" like columns for layered data, or a single "depth" like column for
        discrete data.

        """
        # NOTE: Now more rigid than necessary, we can add more flexibility in the future if needed.
        self._top = "top" if "top" in self._obj.columns else None

        if "depth" in self._obj.columns:
            self._bottom = "depth"
        elif "bottom" in self._obj.columns:
            self._bottom = "bottom"
        else:
            self._bottom = None

    @staticmethod
    def _validate_dataframe(dataframe: DataFrame):
        """
        Check if crucial information is present in a DataFrame to see if methods in the
        accessor can be used.

        Raises
        ------
        KeyError
            If the DataFrame does not contain the required information.

        """
        if "nr" not in dataframe.columns:
            raise KeyError(
                "DataFrame must contain an 'nr' column identifying individual objects."
            )

    @property
    def has_geometry(self):
        """
        Returns True if the object is a GeoDataFrame with a valid geometry column, False
        otherwise. Used to determine whether specfic geospatial operations can be performed
        on the object.

        """
        if isinstance(self._obj, gpd.GeoDataFrame):
            if self._obj._geometry_column_name is not None:
                # Only if self._obj is a GeoDataFrame we can check this attribute
                return True
        return False

    @property
    def has_xy_columns(self):
        """
        Returns True if the object contains x and y columns, False otherwise. Used to
        determine whether specific selection operations can be performed on the object.

        """
        if "x" in self._obj.columns and "y" in self._obj.columns:
            return True
        return False

    @property
    def has_depth_columns(self):
        """
        Returns True if the object contains information about depth, such as 'top' and
        'bottom' or 'depth' columns, False otherwise. Used to determine whether specific
        selection operations can be performed on the object.

        """
        if "surface" in self._obj.columns and self._bottom is not None:
            return True
        return False

    @property
    def _new_survey_id(self):
        """
        Get a boolean mask indicating the locations of new survey IDs in the data. This
        is used to identify the first layer of each survey.

        Returns
        -------
        pd.Series
            Boolean Series with True at locations where a new survey ID starts, and False
            elsewhere.

        """
        return self._obj["nr"] != self._obj["nr"].shift()

    @property
    def _last_row_in_survey(self):
        """
        Get a boolean mask indicating the locations of the last row in each survey. This
        is used to identify the last layer of each survey.

        Returns
        -------
        pd.Series
            Boolean Series with True at locations where the last row of a survey is, and False
            elsewhere.

        """
        return self._obj["nr"] != self._obj["nr"].shift(-1)

    @staticmethod
    def _to_iterable(value: str | Iterable) -> Iterable:
        if isinstance(value, str):
            value = [value]
        return value

    def _get_depth_relative_to_surface(self):
        """
        Get copy of the self._obj with depth columns converted to depth relative
        to surface level.
        """
        data = self._obj.copy()
        data[self._bottom] = data["surface"] - data[self._bottom]
        if self._top:
            data[self._top] = data["surface"] - data[self._top]
        return data

    def validate_with_schema(self, schema: DataFrameSchema):
        """
        Validate the DataFrame using a specified Pandera schema.

        Parameters
        ----------
        schema : DataFrameSchema
            DataFrameSchema to validate the DataFrame with.
        """
        self._obj = validation.safe_validate(self._obj, schema)

    def to_header(
        self,
        include_columns: str | Iterable[str] | None = None,
        exclude_columns: str | Iterable[str] | None = None,
        coordinate_names: tuple[str, str] = None,
        crs: str | int | CRS = None,
    ) -> gpd.GeoDataFrame:
        """
        Create a header GeoDataFrame from the DataFrame to be used as a header table for
        the creation of a :class:`~geost.base.Collection` object. The header contains one
        row per unique object in the DataFrame, identified by the "nr" column. The header
        can contain a geometry column with point geometries created from specified coordinate
        columns in the DataFrame. Optional columns can be given to be included or excluded
        in the header.

        Parameters
        ----------
        include_columns : str | Iterable[str] | None, optional
            Columns to include in the header. The default is None, which includes all
            columns. If "nr" is not included in the specified columns, it will be added
            automatically as it is required.
        exclude_columns : str | Iterable[str] | None, optional
            Columns to exclude from the header. The default is None, which excludes no
            columns. If "nr" is included in the specified columns, an error is raised as
            it is required.
        coordinate_names : tuple[str, str], optional
            Tuple specifying the names of the columns to be used as coordinates for the
            geometry column. The default is None, which means no geometry column will be
            created.
        crs : str | int | CRS, optional
            Coordinate reference system for the geometry column. The default is None,
            which means no CRS will be assigned to the resulting GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing one row per unique object in the DataFrame, with optional
            geometry column and specified columns included or excluded.

        Raises
        ------
        KeyError
            Raised if the specified coordinate columns are not found in the DataFrame.
        ValueError
            Raised if both 'include_columns' and 'exclude_columns' are specified, or if
            'nr' is given in the exclude_columns parameter.

        Examples
        --------
        To create a header GeoDataFrame with all columns from the DataFrame and no geometry
        column:

        >>> data.gst.to_header()

        To create a header GeoDataFrame with only specific columns from the DataFrame and
        no geometry column:

        >>> data.gst.to_header(include_columns=["nr", "column1", "column2"])

        To create a header GeoDataFrame with a geometry column created from specified coordinate
        names, a specific CRS and with columns excluded:

        >>> data.gst.to_header(
        ...     coordinate_names=("x", "y"), crs=28992, exclude_columns=["column1", "column2"]
        ... )

        """
        import shapely

        header = self._obj.drop_duplicates(subset="nr", ignore_index=True)

        if coordinate_names is not None:
            x_col, y_col = coordinate_names
            if not {x_col, y_col}.issubset(header.columns):
                raise KeyError(
                    f"Coordinate columns '{x_col}' and/or '{y_col}' not found in DataFrame."
                )
            geometry = shapely.points(header[[x_col, y_col]])
            # NOTE: future versions may require other geometry types too
        else:
            geometry = None

        if include_columns is not None and exclude_columns is not None:
            raise ValueError(
                "Cannot specify both 'include_columns' and 'exclude_columns'."
            )
        elif include_columns is not None:
            if "nr" not in include_columns:
                include_columns = ["nr"] + list(include_columns)
            header = header[include_columns]
        elif exclude_columns is not None:
            if "nr" in exclude_columns:
                raise ValueError("Cannot exclude 'nr' column from header.")
            header = header.drop(columns=exclude_columns)

        return gpd.GeoDataFrame(header, geometry=geometry, crs=crs)

    def to_collection(
        self,
        has_inclined: bool = False,
        vertical_reference: str | int | CRS = None,
        **header_kwargs,
    ):
        """
        Create a :class:`geost.base.Collection` from the current GeoDataFrame or DataFrame.

        Parameters
        ----------
        has_inclined : bool, optional
            Indicates whether the collection has inclined data. The default is False.
        vertical_reference : str | int | CRS, optional
            Vertical reference system for the collection. The default is None.
        **header_kwargs
            Keyword arguments to be passed to the :class:`geost.accessor.GeostFrame.to_header`
            method for creating the header table.

        Returns
        -------
        :class:`geost.base.Collection`
            A Collection instance created from the current GeoDataFrame or DataFrame.

        """
        from geost.base import Collection  # Avoid circular import

        header = self.to_header(**header_kwargs)
        return Collection(
            self._obj,
            header=header,
            has_inclined=has_inclined,
            vertical_reference=vertical_reference,
        )

    def change_horizontal_reference(self, to_epsg: str | int | CRS) -> None:
        raise NotImplementedError("Method not implemented yet.")

    def change_vertical_reference(
        self, from_epsg: str | int | CRS, to_epsg: str | int | CRS
    ) -> None:
        raise NotImplementedError("Method not implemented yet.")

    @_requires_geometry
    def select_within_bbox(
        self,
        xmin: int | float,
        ymin: int | float,
        xmax: int | float,
        ymax: int | float,
        invert: bool = False,
    ) -> gpd.GeoDataFrame:
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
        invert : bool, optional
            Invert the selection, so select all objects outside of the
            bounding box in this case, by default False.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame instance containing only the selected geometries.

        """
        selection = spatial.select_points_within_bbox(
            self._obj, xmin, ymin, xmax, ymax, invert=invert
        )
        return selection

    @_requires_geometry
    def select_with_points(
        self,
        points: str | Path | gpd.GeoDataFrame | GeometryType,
        max_distance: float | int,
        invert: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Select data based on the distance to given point geometries.

        Parameters
        ----------
        points : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of point geometries that can be used for the selection: GeoDataFrame
            containing points or filepath to a shapefile like file, or Shapely Point,
            MultiPoint or list containing Point objects.
        max_distance : float | int
            Maximum distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame instance containing only the selected geometries.

        """
        selection = spatial.select_points_near_points(
            self._obj, points, max_distance, invert=invert
        )
        return selection

    @_requires_geometry
    def select_with_lines(
        self,
        lines: str | Path | gpd.GeoDataFrame | GeometryType,
        max_distance: float | int,
        invert: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Select data based on the distance to given line geometries.

        Parameters
        ----------
        lines : str | Path | gpd.GeoDataFrame | GeometryType
            Any type of line geometries that can be used for the selection: GeoDataFrame
            containing lines or filepath to a shapefile like file, or Shapely LineString,
            MultiLineString or list containing LineString objects.
        max_distance : float | int
            Maximum distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame instance containing only the selected geometries.

        """
        selection = spatial.select_points_near_lines(
            self._obj, lines, max_distance, invert=invert
        )
        return selection

    @_requires_geometry
    def select_within_polygons(
        self,
        polygons: str | Path | gpd.GeoDataFrame | GeometryType,
        buffer: float | int = 0,
        invert: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Select data based on polygon geometries.

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
        gpd.GeoDataFrame
            GeoDataFrame instance containing only the selected geometries.

        """
        selection = spatial.select_points_within_polygons(
            self._obj, polygons, buffer, invert=invert
        )
        return selection

    def select_by_depth(
        self,
        top_min: float = None,
        top_max: float = None,
        end_min: float = None,
        end_max: float = None,
    ) -> gpd.GeoDataFrame:
        raise NotImplementedError("Method not implemented yet.")

    @_requires_geometry
    def get_area_labels(
        self,
        polygon_gdf: str | Path | gpd.GeoDataFrame,
        column_name: str | Iterable,
        include_in_header=False,
    ) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented yet.")

    def select_by_values(
        self,
        column: str,
        values: str | Iterable | slice,
        how: str = "or",
        invert: bool = False,
        inclusive: str = "both",
    ) -> pd.DataFrame:
        """
        Select data based on the presence of given values in a given column. Can be used
        for example to select data that contain peat in the lithology column.

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
        pd.DataFrame
            New DataFrame containing only the data selected by this method.

        Examples
        --------
        To select data where both clay ("K") and peat ("V") are present at the same
        time, use "and" as a selection method:

        >>> data.gst.select_by_values("lith", ["V", "K"], how="and")

        To select data that can have one, or both lithologies, use or as the selection
        method:

        >>> data.gst.select_by_values("lith", ["V", "K"], how="or")

        In case of numerical values, use a slice to select data that contain a specific
        range of values. For example, to select data that contain cone resistances ("qc")
        between 15 and 20 MPa:

        >>> data.gst.select_by_values("qc", slice(15, 20))

        """
        if column not in self._obj.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        selected = self._select_by_values(values, column, how, inclusive)

        if invert:
            exclude_indices = selected.index
            selected = self._obj.loc[~self._obj.index.isin(exclude_indices)]

        return selected

    @singledispatchmethod
    def _select_by_values(
        self, values: Any, column: str, how: str, inclusive: str
    ) -> pd.DataFrame:
        raise TypeError(
            f"Unsupported type of selection values: {type(values)} values must be "
            "either a string\n, an iterable, or a slice for numerical values."
        )

    @_select_by_values.register(str)
    def _(self, values, column, *_) -> pd.DataFrame:
        selected = self._obj
        valid = selected["nr"][selected[column] == values].unique()
        selected = selected[selected["nr"].isin(valid)]
        return selected

    @_select_by_values.register(
        set | list | np.ndarray | pd.Series | pd.arrays.ArrowStringArray
    )
    def _(self, values, column, how, _) -> pd.DataFrame:
        selected = self._obj
        if how == "or":
            valid = selected["nr"][selected[column].isin(values)].unique()
            selected = selected[selected["nr"].isin(valid)]

        elif how == "and":
            for value in values:
                valid = selected["nr"][selected[column] == value].unique()
                selected = selected[selected["nr"].isin(valid)]
        return selected

    @_select_by_values.register(slice)
    def _(self, values, column, _, inclusive) -> pd.DataFrame:
        if not pd.api.types.is_numeric_dtype(self._obj[column]):
            raise TypeError("Can only use a slice selection on numerical columns.")

        selected = self._obj
        valid = selected["nr"][
            selected[column].between(values.start, values.stop, inclusive)
        ].unique()
        selected = selected[selected["nr"].isin(valid)]
        return selected

    @_requires_depth
    def slice_depth_interval(
        self,
        upper_boundary: float | int = None,
        lower_boundary: float | int = None,
        relative_to_vertical_reference: bool = False,
        update_layer_boundaries: bool = True,
    ) -> pd.DataFrame:
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
            Only used when the data is layered, i.e. the data contains tops and bottoms for
            individual layers. If True, the layer boundaries in the sliced data are updated
            according to the upper and lower boundaries used with the slice. If False, the
            original layer boundaries are kept in the sliced object. The default is False.

        Returns
        -------
        pd.DataFrame
            New DataFrame containing only the data selected by this method.

        Examples
        --------
        Usage depends on whether the slicing is done with respect to depth below the
        surface or to a vertical reference plane.

        For example, select layers in data that are between 2 and 3 meters below the
        surface:

        >>> data.gst.slice_depth_interval(2, 3)

        By default, the method updates the layer boundaries in sliced object according to
        the upper and lower boundaries. To suppress this behaviour use:

        >>> data.gst.slice_depth_interval(2, 3, update_layer_boundaries=False)

        Slicing can also be done with respect to a vertical reference plane like "NAP".
        For example, to select layers in boreholes that are between -3 and -5 m NAP, use:

        >>> data.gst.slice_depth_interval(-3, -5, relative_to_vertical_reference=True)

        """
        sliced = self._obj

        if not upper_boundary:
            upper_boundary = 1e34 if relative_to_vertical_reference else -1e34

        if not lower_boundary:
            lower_boundary = -1e34 if relative_to_vertical_reference else 1e34

        if relative_to_vertical_reference:
            upper_boundary = self._obj["surface"] - upper_boundary
            lower_boundary = self._obj["surface"] - lower_boundary

        if layered_selection := self._top and self._bottom:
            sliced = sliced[
                (sliced[self._bottom] > upper_boundary)
                & (sliced[self._top] < lower_boundary)
            ]
        else:
            sliced = sliced[
                sliced[self._bottom].between(upper_boundary, lower_boundary)
            ]

        # We do not update layer boundaries when slicing discrete data
        if update_layer_boundaries and layered_selection:
            bounds_are_series = True if relative_to_vertical_reference else False
            if bounds_are_series:
                upper_boundary = upper_boundary.loc[sliced.index]
                lower_boundary = lower_boundary.loc[sliced.index]

            sliced.loc[sliced[self._top] <= upper_boundary, self._top] = upper_boundary
            sliced.loc[sliced[self._bottom] >= lower_boundary, self._bottom] = (
                lower_boundary
            )

        return sliced

    def slice_by_values(
        self,
        column: str,
        values: str | Iterable | slice,
        invert: bool = False,
        inclusive: str = "both",
    ) -> pd.DataFrame:
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
        pd.DataFrame
            New DataFrame containing only the data objects selected by this method.

        Examples
        --------
        Return only rows in data contain sand ("Z") as lithology:

        >>> data.gst.slice_by_values("lith", "Z")

        If you want all the rows that may contain everything but sand, use the "invert"
        option:

        >>> data.gst.slice_by_values("lith", "Z", invert=True)

        If you want to slice a range of numerical values, you can use a slice. For example,
        to return only rows where the cone resistance ("qc") is between 15 and 20 MPa:

        >>> data.gst.slice_by_values("qc", slice(15, 20))

        """
        sliced = self._slice_by_values(values, column, inclusive)

        if invert:
            exclude_indices = sliced.index
            sliced = self._obj.loc[~self._obj.index.isin(exclude_indices)]

        return sliced

    @singledispatchmethod
    def _slice_by_values(
        self, values: Any, column: str, inclusive: str
    ) -> pd.DataFrame:
        raise TypeError(f"Unsupported type of selection values: {type(values)}")

    @_slice_by_values.register(str)
    def _(self, values, column, _) -> pd.DataFrame:
        return self._obj[self._obj[column] == values]

    @_slice_by_values.register(set | list | np.ndarray | pd.Series)
    def _(self, values, column, _) -> pd.DataFrame:
        return self._obj[self._obj[column].isin(values)]

    @_slice_by_values.register(slice)
    def _(self, values, column, inclusive) -> pd.DataFrame:
        if not pd.api.types.is_numeric_dtype(self._obj[column]):
            raise TypeError("Can only use a slice selection on numerical columns.")

        return self._obj[
            self._obj[column].between(values.start, values.stop, inclusive)
        ]

    def select_by_condition(self, condition: Any, invert: bool = False) -> pd.DataFrame:
        """
        Select data using a manual condition that results in a boolean mask. Returns the
        rows in the data where the 'condition' evaluates to True.

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
        pd.DataFrame
            New DataFrame containing only the data selected by this method.

        Examples
        --------
        Select rows in data that contain a specific value:

        >>> data.gst.select_by_condition(data["lith"]=="V")

        Or select rows in the data that contain a specific (part of) string or strings:

        >>> data.gst.select_by_condition(data["column"].str.contains("foo|bar"))

        """
        if invert:
            selected = self._obj[~condition]
        else:
            selected = self._obj[condition]

        return selected

    @_requires_depth
    def calculate_thickness(self) -> pd.Series:
        """
        Calculate the thickness of layers in the data. This method requires the presence
        of depth information in the data. See the `has_depth_columns` property for more
        information on what kind of depth columns are required.

        Returns
        -------
        pd.Series
            Series containing the thickness for each layer in the data.

        """
        if self._top and self._bottom:
            thickness = (self._obj[self._top] - self._obj[self._bottom]).abs()
        else:
            thickness = self._obj[self._bottom].diff()

            thickness[self._new_survey_id] = self._obj[
                self._bottom
            ]  # Thickness of first layer is equal to depth of first layer data is discrete

        return thickness

    @_requires_depth
    def get_cumulative_thickness(
        self, column: str, values: str | list[str] | slice
    ) -> pd.Series:
        """
        Get the cumulative thickness of layers where a column contains a specified search
        value or values, or falls within a specified range.

        Parameters
        ----------
        column : str
            Name of column that must contain the search value or values.
        values : str | List[str] | slice
            Search value or values in the column to find the cumulative thickness for. In
            case of numerical values, a slice can be used to specify a range of values to
            search for, see example below.

        Returns
        -------
        pd.Series
            Series containing the cumulative thickness for each survey "nr" in the data.

        Examples
        --------
        Get the cumulative thickness of the layers with lithology "K" in the column "lith"
        use:

        >>> data.gst.get_cumulative_thickness("lith", "K")

        Or get the cumulative thickness for multiple selection values. This calculates
        the cumulative thickness of the combined values:

        >>> data.gst.get_cumulative_thickness("lith", ["K", "Z"])

        In case of numerical values, a slice can be used to specify a range of values to
        search for. For example, to get the cumulative thickness of layers with a cone
        resistance ("qc") between 15 and 20 MPa:

        >>> data.gst.get_cumulative_thickness("qc", slice(15, 20))

        """
        selected_layers = self.slice_by_values(column, values)

        if "thickness" not in self._obj.columns:
            thickness = self.calculate_thickness()
            selected_layers["thickness"] = thickness.loc[selected_layers.index]

        grouped = selected_layers.groupby("nr", sort=False)
        thickness = grouped["thickness"].sum()
        return thickness

    @_requires_depth
    def get_layer_top(
        self, column: str, values: str | list[str] | slice, min_thickness: float = None
    ) -> pd.Series:
        selection = self.slice_by_values(column, values)

        if "thickness" not in self._obj.columns:
            thickness = self.calculate_thickness()
            selection["thickness"] = thickness.loc[selection.index]

        if min_thickness is not None:
            selection = selection[selection["thickness"] >= min_thickness]

        if self._top is None:
            selection[self._bottom] = selection[self._bottom] - selection["thickness"]
            # In case of discrete data, we calculate the top of the layer because the depth
            # is not the actual top of the layer but its base.
            layer_top = selection.groupby("nr", sort=False)[self._bottom].first()
        else:
            layer_top = selection.groupby("nr", sort=False)[self._top].first()

        return layer_top

    @_requires_depth
    def get_layer_base(
        self, column: str, values: str | list[str] | slice, min_thickness: float = None
    ) -> pd.Series:
        selection = self.slice_by_values(column, values)

        if "thickness" not in self._obj.columns:
            thickness = self.calculate_thickness()
            selection["thickness"] = thickness.loc[selection.index]

        if min_thickness is not None:
            selection = selection[selection["thickness"] >= min_thickness]

        grouped = selection.groupby("nr", sort=False)
        layer_base = grouped[self._bottom].last()
        return layer_base

    @_requires_depth
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
        self._check_has_xy()
        data = self._get_depth_relative_to_surface()
        displayed_variables = self._to_iterable(displayed_variables)

        if self._top:
            vtk_object = export.borehole_to_multiblock(
                data,
                [self._top, self._bottom],
                displayed_variables,
                radius,
                n_sides,
                vertical_factor,
            )
        else:
            vtk_object = export.borehole_to_multiblock(
                data,
                self._bottom,
                displayed_variables,
                radius,
                n_sides,
                vertical_factor,
            )

        return vtk_object

    @_requires_depth
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
        self._check_has_xy()
        data = self._get_depth_relative_to_surface()
        displayed_variables = self._to_iterable(displayed_variables)

        if self._top:
            vtk_object = export.layerdata_to_pyvista_unstructured(
                data,
                [self._top, self._bottom],
                displayed_variables,
                radius=radius,
            )
        else:
            vtk_object = export.layerdata_to_pyvista_unstructured(
                data, self._bottom, displayed_variables, radius=radius
            )

        return vtk_object

    @_requires_depth
    def to_datafusiontools(
        self,
        columns: list[str],
        outfile: str | Path = None,
        encode: bool = False,
        relative_to_vertical_reference: bool = True,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def _create_geodataframe_3d(
        self,
        crs: str | int | CRS = None,
    ):
        """
        Helper method for export method "to_qgis3d" to create the necessary GeoDataFrame
        containing 3D Shapely objects and associated information.

        Parameters
        ----------
        relative_to_vertical_reference : bool, optional
            If True, the depth of all data objects will converted to a depth with respect to
            a reference plane (e.g. "NAP", "TAW"). If False, the depth will be kept as original
            in the "top" and "bottom" columns which is in meter below the surface. The default
            is True.
        crs : str | int | CRS
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input().
        has_inclined : bool, optional
            If True, the data also contains inclined objects which means the top of layers
            is not in the same x,y-location as the bottom of layers. The default is False.

        """
        data = self._get_depth_relative_to_surface()

        data_columns = [
            col
            for col in data.columns
            if col not in {"nr", "x", "y", "surface", "depth", "top", "bottom"}
        ]

        # Prepare top, bottom and corresponding x and y values for geometry creation.
        # Infer the top of layers from the bottom of layers if the top column is not present.
        if self._top is None:
            top = data[self._bottom].shift().mask(self._new_survey_id, data["surface"])
            x_bot, y_bot = data["x"], data["y"]
            x_top = data["x"].shift().mask(self._new_survey_id, data["x"])
            y_top = data["y"].shift().mask(self._new_survey_id, data["y"])
        else:
            top = data[self._top]
            x_top, y_top = data["x"], data["y"]
            x_bot = data["x"].shift(-1).mask(self._last_row_in_survey, data["x"])
            y_bot = data["y"].shift(-1).mask(self._last_row_in_survey, data["y"])

        # TODO: slight problem persists here for inclined boreholes. The last layer will
        # always be non-inclined because we miss information on the bottom x/y coords.
        # This cannot be correctly inferred from the data.

        data_to_write = dict(
            nr=data["nr"].values,
            top=top.values.astype(float),
            bottom=data[self._bottom].values.astype(float),
        )

        data_to_write.update(data[data_columns].to_dict(orient="list"))

        geometries = linestrings(
            [
                [[x, y, top + 0.01], [x_bot, y_bot, bottom + 0.01]]
                for x, y, x_bot, y_bot, top, bottom in zip(
                    x_top.values.astype(float),
                    y_top.values.astype(float),
                    x_bot.values.astype(float),
                    y_bot.values.astype(float),
                    top.values.astype(float),
                    data[self._bottom].values.astype(float),
                )
            ]
        )

        return gpd.GeoDataFrame(
            data=data_to_write,
            geometry=geometries,
            crs=crs,
        )

    def to_qgis3d(
        self,
        outfile: str | Path,
        crs: str | int | CRS = None,
        **kwargs,
    ):
        """
        Write data to geopackage file that can be directly loaded in the Qgis2threejs
        plugin. Works only for layered (borehole) data.

        Parameters
        ----------
        outfile : str | Path
            Path to geopackage file to be written.
        crs : str | int | CRS
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input().

        **kwargs
            geopandas.GeodataFrame.to_file kwargs. See relevant Geopandas documentation.

        """
        self._check_has_depth()
        self._check_has_xy()
        qgis3d = self._create_geodataframe_3d(crs=crs)
        qgis3d.to_file(outfile, driver="GPKG", **kwargs)

    def to_kingdom(
        self,
        outfile: str | Path,
        tdstart: int = 1,
        vw: float = 1500.0,
        vs: float = 1600.0,
    ):
        """
        Write data to 2 csv files: 1) interval data and 2) time-depth chart. These files
        can be imported in the Kingdom seismic interpretation software.

        Parameters
        ----------
        outfile : str | Path
            Path to csv file to be written.
        tdstart : int
            startindex for TDchart, default is 1
        vw : float
            sound velocity in water in m/s, default is 1500 m/s
        vs : float
            sound velocity in sediment in m/s, default is 1600 m/s

        """
        self._check_has_depth()
        self._check_has_xy()

        # 1. add columns needed in kingdom and write interval data
        kingdom_df = self._get_depth_relative_to_surface()

        if self._top is None:
            kingdom_df["top"] = (
                kingdom_df[self._bottom]
                .shift()
                .mask(self._new_survey_id, kingdom_df["surface"])
            )

        # Add total depth NOTE: is the insert at idx 7 really necessary?
        kingdom_df.insert(
            7,
            "Total depth",
            (kingdom_df["surface"] - kingdom_df[self._bottom])[
                self._last_row_in_survey
            ],
        )
        kingdom_df["Total depth"] = kingdom_df["Total depth"].bfill()

        # Rename bottom and top columns to Kingdom requirements
        kingdom_df.rename(
            columns={"top": "Start depth", self._bottom: "End depth"}, inplace=True
        )
        kingdom_df.to_csv(outfile, index=False)

        # 2. create and write time-depth chart
        tdchart = self._obj[["nr", "surface"]]
        tdchart.drop_duplicates(inplace=True)
        tdchart.insert(0, "id", range(tdstart, tdstart + len(tdchart)))

        # Add measured depth (predefined depths of 0 and 1 m below surface)
        tdchart = pd.concat(
            [
                tdchart.assign(MD=np.zeros(len(tdchart), dtype=np.int64)),
                tdchart.assign(MD=np.ones(len(tdchart), dtype=np.int64)),
            ]
        )
        # Add two-way travel time
        tdchart["TWT"] = (-tdchart["surface"] / (vw / 2 / 1000)) + (
            tdchart["MD"] * 1 / (vs / 2 / 1000)
        )

        tdchart.drop("surface", axis=1, inplace=True)
        tdchart.sort_values(by=["id", "MD"], inplace=True)
        if not isinstance(outfile, Path):
            outfile = Path(outfile)
        tdchart.to_csv(
            outfile.parent.joinpath(f"{outfile.stem}_TDCHART{outfile.suffix}"),
            index=False,
        )
