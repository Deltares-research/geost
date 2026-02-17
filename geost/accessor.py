from typing import TYPE_CHECKING, Any, Iterable, List

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import buffer

from geost import spatial

if TYPE_CHECKING:
    from pathlib import Path

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
    def _validate_dataframe(dataframe):
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
    def has_depth_columns(self):
        """
        Returns True if the object contains information about depth, such as 'top' and
        'bottom' or 'depth' columns, False otherwise. Used to determine whether specific
        selection operations can be performed on the object.

        """
        if "surface" in self._obj.columns and self._bottom is not None:
            return True

        return False

    @staticmethod
    def _to_iterable(selection_values: str | Iterable) -> Iterable:
        if isinstance(selection_values, str):
            selection_values = [selection_values]
        return selection_values

    def _check_has_geometry(self):
        if not self.has_geometry:
            raise TypeError(
                "Object is not a GeoDataFrame with a valid geometry column."
            )

    def _check_has_depth(self):
        if not self.has_depth_columns:
            raise KeyError(  # TODO: Check formatting of this error message
                "Object does not contain depth information needed 'surface' and 'depth'\n"
                "columns. One of the following combinations of columns is required:\n"
                " - 'surface', 'top' and 'bottom'\n"
                " - 'surface' and 'bottom'\n"
                " - 'surface' and 'depth'\n"
            )

    def change_horizontal_reference(self, to_epsg: str | int | CRS) -> None:
        raise NotImplementedError("Method not implemented yet.")

    def change_vertical_reference(
        self, from_epsg: str | int | CRS, to_epsg: str | int | CRS
    ) -> None:
        raise NotImplementedError("Method not implemented yet.")

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
        self._check_has_geometry()
        selection = spatial.select_points_within_bbox(
            self._obj, xmin, ymin, xmax, ymax, invert=invert
        )
        return selection

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
        self._check_has_geometry()
        selection = spatial.select_points_near_points(
            self._obj, points, max_distance, invert=invert
        )
        return selection

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
        self._check_has_geometry()
        selection = spatial.select_points_near_lines(
            self._obj, lines, max_distance, invert=invert
        )
        return selection

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
        self._check_has_geometry()
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

    def get_area_labels(
        self,
        polygon_gdf: str | Path | gpd.GeoDataFrame,
        column_name: str | Iterable,
        include_in_header=False,
    ) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented yet.")

    def to_header(
        self, horizontal_reference: str | int | CRS = 28992
    ) -> gpd.GeoDataFrame:
        raise NotImplementedError("Method not implemented yet.")

    def to_collection(
        self,
        has_inclined: bool = False,
        horizontal_reference: str | int | CRS = 28992,
        vertical_reference: str | int | CRS = 5709,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def select_by_values(
        self, column: str, selection_values: str | Iterable, how: str = "or"
    ) -> pd.DataFrame:
        """
        Select data based on the presence of given values in a given column. Can be used
        for example to select data that contain peat in the lithology column.

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

        """
        if column not in self._obj.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        selection_values = self._to_iterable(selection_values)

        selected = self._obj
        if how == "or":
            valid = self._obj["nr"][self._obj[column].isin(selection_values)].unique()
            selected = selected[selected["nr"].isin(valid)]

        elif how == "and":
            for value in selection_values:
                valid = self._obj["nr"][self._obj[column] == value].unique()
                selected = selected[selected["nr"].isin(valid)]

        return selected

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
        self._check_has_depth()
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
        self, column: str, selection_values: str | Iterable, invert: bool = False
    ) -> pd.DataFrame:
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
        pd.DataFrame
            New DataFrame containing only the data objects selected by this method.

        Examples
        --------
        Return only rows in data contain sand ("Z") as lithology:

        >>> data.gst.slice_by_values("lith", "Z")

        If you want all the rows that may contain everything but sand, use the "invert"
        option:

        >>> data.gst.slice_by_values("lith", "Z", invert=True)

        """
        selection_values = self._to_iterable(selection_values)

        sliced = self._obj

        if invert:
            sliced = sliced[~sliced[column].isin(selection_values)]
        else:
            sliced = sliced[sliced[column].isin(selection_values)]

        return sliced

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

    def get_cumulative_thickness(
        self, column: str, values: str | List[str] | slice
    ) -> pd.Series:
        """
        Get the cumulative thickness of layers where a column contains a specified search
        value or values.

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
        self._check_has_depth()

        if isinstance(values, slice):
            search_mask = self._obj[column].between(values.start, values.stop)
            selected_layers = self._obj[search_mask]
        else:
            selected_layers = self.slice_by_values(column, values)

        grouped = selected_layers.groupby("nr")
        if self._top and self._bottom:
            thickness = grouped.apply(
                lambda df: np.abs(df[self._top] - df[self._bottom]).sum()
            )
        else:
            selected_layers["thickness"] = grouped[self._bottom].diff()
            selected_layers["thickness"] = selected_layers["thickness"].fillna(
                grouped["thickness"].transform("mean")
            )
            thickness = grouped["thickness"].sum()

        return thickness

    def get_layer_top(
        self,
        column: str,
        values: str | List[str],
        min_thickness: float = 0,
        min_depth: float = 0,
    ) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented yet.")

    def get_layer_base(
        self,
        column: str,
        values: str | List[str],
        min_thickness: float = 0,
        min_depth: float = 0,
    ) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented yet.")

    def to_pyvista_cylinders(
        self,
        displayed_variables: str | List[str],
        radius: float = 1,
        vertical_factor: float = 1.0,
        relative_to_vertical_reference: bool = True,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def to_pyvista_grid(
        self,
        displayed_variables: str | list[str],
        radius: float = 1,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def to_datafusiontools(
        self,
        columns: List[str],
        outfile: str | Path = None,
        encode: bool = False,
        relative_to_vertical_reference: bool = True,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def _create_geodataframe_3d(
        self,
        relative_to_vertical_reference: bool = True,
        crs: str | int | CRS = None,
        has_inclined: bool = False,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def to_qgis3d(
        self,
        outfile: str | Path,
        relative_to_vertical_reference: bool = True,
        crs: str | int | CRS = None,
        has_inclined: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("Method not implemented yet.")

    def to_kingdom(
        self,
        outfile: str | Path,
        tdstart: int = 1,
        vw: float = 1500.0,
        vs: float = 1600.0,
    ):
        raise NotImplementedError("Method not implemented yet.")
