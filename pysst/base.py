import pandas as pd
from pathlib import WindowsPath
from typing import List, Union, TypeVar, Iterable, Optional
from functools import reduce

# Local imports
from pysst import spatial
from pysst.export import borehole_to_multiblock
from pysst.utils import MissingOptionalModule
from pysst.analysis import cumulative_thickness, layer_top

# Optional imports
try:
    import geopandas as gpd

    create_header = spatial.header_to_geopandas
except:
    gpd = MissingOptionalModule("geopandas")
    def create_header(x): return x


Coordinate = TypeVar("Coordinate", int, float)
GeoDataFrame = TypeVar("GeoDataFrame")


class PointDataCollection:
    """
    Dataclass for collections of pointdata, such as boreholes and CPTs. The pysst module
    revolves around this class and includes all methods that apply generically to both
    borehole and CPT data, such as selection and export methods.

    Args:
        data (pd.DataFrame): Dataframe containing borehole/CPT data.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        vertical_reference: str,
        header: Optional[pd.DataFrame] = None,
    ):
        self.data = data
        if isinstance(header, pd.DataFrame):
            self.header = header
        else:
            self.reset_header()
        self.__vertical_reference = vertical_reference

    def __new__(cls, *args, **kwargs):
        if cls is PointDataCollection:
            raise TypeError(
                f"Cannot construct {cls.__name__} directly: construct class from its children instead"
            )
        else:
            return object.__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n# header = {self.n_points}"

    def reset_header(self):
        header = self.data.drop_duplicates(subset="nr")
        header = header[["nr", "x", "y", "mv", "end"]].reset_index(drop=True)
        self.header = create_header(header)

    @property
    def header(self):
        """
        This attribute is a dataframe of header (1 row per borehole/cpt) and includes:
        point id, x-coordinate, y-coordinate, surface level and end depth:

        Column names:
        ["nr", "x", "y", "mv", "end"]
        """
        return self._header

    @property
    def data(self):
        return self._data

    @property
    def n_points(self):
        return len(self.header)

    @property
    def vertical_reference(self):
        return self.__vertical_reference

    @header.setter
    def header(self, header):
        # Whenever self.header is set (inside a method or through an instance),
        # this setter method is called. Hence calling the TODO validation is only
        # required here. Currently there is no validation, so the header can be set to
        # anything by accident.

        # validate_header(input)
        self._header = header

    @data.setter
    def data(self, data):
        # Same as for the header the setter is always called when trying to set
        # self.data. TODO Data validation is applied here and upon pass the protected
        # attr self._data is set.

        # validate_data(input)
        self._data = data

    def change_vertical_reference(self, to: str):
        """
        Change the vertical reference of layer tops and bottoms

        Parameters
        ----------
        to : str
            To which vertical reference to convert the layer tops and bottoms. Either
            'NAP', 'surfacelevel' or 'depth'.

            NAP = elevation with respect to NAP datum.
            surfacelevel = elevation with respect to surface (surface is 0 m, e.g.
                           layers tops could be 0, -1, -2 etc.).
            depth = depth with respect to surface (surface is 0 m, e.g. depth of layers
                    tops could be 0, 1, 2 etc.).
        """
        match self.__vertical_reference:
            case "NAP":
                if to == "surfacelevel":
                    self._data["top"] = self._data["top"] - self._data["mv"]
                    self._data["bottom"] = self._data["bottom"] - \
                        self._data["mv"]
                    self.__vertical_reference = "surfacelevel"
                elif to == "depth":
                    self._data["top"] = (
                        self._data["top"] - self._data["mv"]) * -1
                    self._data["bottom"] = (
                        self._data["bottom"] - self._data["mv"]
                    ) * -1
                    self.__vertical_reference = "depth"
            case "surfacelevel":
                if to == "NAP":
                    self._data["top"] = self._data["top"] + self._data["mv"]
                    self._data["bottom"] = self._data["bottom"] + \
                        self._data["mv"]
                    self.__vertical_reference = "NAP"
                if to == "depth":
                    self._data["top"] = self._data["top"] * -1
                    self._data["bottom"] = self._data["bottom"] * -1
                    self.__vertical_reference = "depth"
            case "depth":
                if to == "NAP":
                    self._data["top"] = self._data["top"] * - \
                        1 + self._data["mv"]
                    self._data["bottom"] = self._data["bottom"] * - \
                        1 + self._data["mv"]
                    self.__vertical_reference = "NAP"
                if to == "surfacelevel":
                    self._data["top"] = self._data["top"] * -1
                    self._data["bottom"] = self._data["bottom"] * -1
                    self.__vertical_reference = "surfacelevel"

    def select_within_bbox(
        self,
        xmin: Coordinate,
        xmax: Coordinate,
        ymin: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):
        """
        Make a selection of the data based on a bounding box.

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
        Child of PointDataCollection.
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = spatial.header_from_bbox(
            self.header, xmin, xmax, ymin, ymax, invert
        )
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_with_points(
        self,
        point_gdf: GeoDataFrame,
        buffer: float = 100,
        invert: bool = False,
    ):
        """
        Make a selection of the data based on points

        Parameters
        ----------
        line_file : Union[str, WindowsPath]
            Shapefile or geopackage containing point data
        buffer: float, default 100
            Buffer around the lines to select points. Default 100
        invert: bool, default False
            Invert the selection

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = spatial.header_from_points(
            self.header, point_gdf, buffer, invert
        )
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_with_lines(
        self,
        line_gdf: GeoDataFrame,
        buffer: float = 100,
        invert: bool = False,
    ):
        """
        Make a selection of the data based on lines

        Parameters
        ----------
        line_file : Union[str, WindowsPath]
            Shapefile or geopackage containing linestring data
        buffer: float, default 100
            Buffer around the lines to select points. Default 100
        invert: bool, default False
            Invert the selection

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = spatial.header_from_lines(
            self.header, line_gdf, buffer, invert
        )
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_within_polygons(
        self,
        polygon_gdf: GeoDataFrame,
        buffer: float = 0,
        invert: bool = False,
    ):
        """
        Make a selection of the data based on polygons

        Parameters
        ----------
        polygon_file : Union[str, WindowsPath]
            Shapefile or geopackage containing (multi)polygon data
        invert: bool, default False
            Invert the selection

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = spatial.header_from_polygons(
            self.header, polygon_gdf, buffer, invert
        )
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_by_values(
        self, column: str, selection_values: Union[str, Iterable], how: str = "or"
    ):
        """
        Select pointdata based on the presence of given values in the given columns. Can be used for example
        to return a BoreholeCollection of boreholes that contain peat in the lithology column. This can be achieved
        by passing e.g. the following arguments to the method:

        self.select_by_values("lith", ["V", "K"], how="and") - Returns boreholes where lithoclasses "V" and "K" are present at the same time

        self.select_by_values("lith", ["V", "K"], how="or") - Returns boreholes where either lithoclasses "V" or "K" are present (or both by coincidence)

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for values.
        selection_values : Union[str, Iterable]
            Values to look for in the column
        how : str
            Either "and" or "or". "and" requires all selction values to be present in column for selection. "or" will select the core if any one
            of the selection_values are found in the column. Default is "and"

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        if not column in self.data.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        if isinstance(selection_values, str):
            selection_values = [selection_values]

        header_copy = self.header.copy()
        if how == "and":
            notna = self.data["nr"][self.data[column].isin(
                selection_values)].unique()
            selected_header = header_copy[header_copy["nr"].isin(notna)]
        elif how == "and":
            for selection_value in selection_values:
                notna = self.data["nr"][
                    self.data[column].isin([selection_value])
                ].unique()
                subselections.append(
                    header_copy[header_copy["nr"].isin(notna)])
            selected_header = pd.concat(subselections)

        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_by_depth(
        self,
        top_min: float = None,
        top_max: float = None,
        end_min: float = None,
        end_max: float = None,
    ):
        """
        Select data from depth constraints. If a keyword argument is not given it will not be considered.
        e.g. if you need only boreholes that go deeper than -500 m use only end_max = -500

        Parameters
        ----------
        top_min : float, optional
            Minimum elevation of the borehole/cpt top, by default None
        top_max : float, optional
            Maximum elevation of the borehole/cpt top, by default None
        end_min : float, optional
            Minimum elevation of the borehole/cpt end, by default None
        end_max : float, optional
            Maximumelevation of the borehole/cpt end, by default None

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
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

        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_by_length(self, min_length: float = None, max_length: float = None):
        """
        Select data from length constraints: e.g. all boreholes between 50 and 150 m long.
        If a keyword argument is not given it will not be considered.

        Parameters
        ----------
        min_length : float, optional
            Minimum length of borehole/cpt, by default None
        max_length : float, optional
            Maximum length of borehole/cpt, by default None

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = self.header.copy()
        length = selected_header["mv"] - selected_header["end"]
        if min_length is not None:
            selected_header = selected_header[length >= min_length]
        if max_length is not None:
            selected_header = selected_header[length <= max_length]

        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def get_area_labels(
        self, polygon_gdf: GeoDataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Find in which area (polygons) the point data locations fall. e.g. to determine in which
        geomorphological unit points are located

        Parameters
        ----------
        polygon_gdf : gpd.GeoDataFrame
            GeoDataFrame with polygons
        column_name : str
            The column name to find the labels in

        Returns
        -------
        pd.DataFrame
            Borehole ids and the polygon label they are in
        """
        area_labels = spatial.find_area_labels(
            self.header, polygon_gdf, column_name)

        return area_labels

    def get_cumulative_layer_thickness(
        self, column: str, values: Union[str, List[str]], include_in_header=False
    ):
        """
        Get the cumulative thickness of layers of a certain type.

        For example, to get the cumulative thickness of the layers with lithology "K" in the column "lith" use:

        self.get_cumulative_layer_thickness("lith", "K")

        Parameters
        ----------
        column : str
            Name of column that contains categorical data
        values : str or List[str]
            Value(s) of entries in column that you want to find the cumulative thickness of
        include_in_header :
            Whether to add the acquired data to the header table or not, By default False
        """
        if isinstance(values, str):
            values = [values]

        result_dfs = []
        for value in values:
            cumulative_thicknesses = cumulative_thickness(
                self.data, column, value)
            result_df = pd.DataFrame(
                cumulative_thicknesses, columns=("nr", f"{value}_thickness")
            )
            result_dfs.append(result_df)

        result = reduce(lambda left, right: pd.merge(
            left, right, on="nr"), result_dfs)
        if include_in_header:
            self._header = self.header.merge(result, on="nr")
        else:
            return result

    def get_layer_top(
        self, column: str, values: Union[str, List[str]], include_in_header=False
    ):
        """
        Find the depth at which a specified layer first occurs

        Parameters
        ----------
        column : str
            Name of column that contains categorical data
        value : str
            Value of entries in column that you want to find top of
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default False
        """
        if isinstance(values, str):
            values = [values]

        result_dfs = []
        for value in values:
            layer_tops = layer_top(self.data, column, value)
            result_df = pd.DataFrame(
                layer_tops, columns=("nr", f"{value}_top"))
            result_dfs.append(result_df)

        result = reduce(lambda left, right: pd.merge(
            left, right, on="nr"), result_dfs)
        if include_in_header:
            self._header = self.header.merge(result, on="nr")
        else:
            return result

    def append(self, other):
        """
        Append data of other object of the same type (e.g BoreholeCollection to BoreholeCollection).

        Parameters
        ----------
        other : Instance of the same type as self.
            Another object of the same type, from which the data is appended to self.
        """
        if self.__class__ == other.__class__:
            # Check overlap first and remove duplicates from 'other' if required
            other_header_overlap = other.header["nr"].isin(self.header["nr"])
            if any(other_header_overlap):
                other_header = other.header[~other_header_overlap]
                other_data = other.data.loc[other.data["nr"].isin(
                    other_header["nr"])]
            else:
                other_header = other.header
                other_data = other.data

            self._data = pd.concat([self.data, other_data])
            self._header = pd.concat([self.header, other_header])
            # TODO return newly constructed class from header and data combination
            # once class construction from header and data is implemented.
        else:
            raise TypeError(
                f"Cannot join instance of {self.__class__} with an instance of {other.__class__}"
            )

    def to_parquet(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write data to parquet file.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to parquet file to be written.
        **kwargs
            pd.DataFrame.to_parquet kwargs.
        """
        self._data.to_parquet(out_file, **kwargs)

    def to_csv(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write data to csv file.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to csv file to be written.
        **kwargs
            pd.DataFrame.to_csv kwargs.
        """
        self._data.to_csv(out_file, **kwargs)

    def to_shape(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write header data to shapefile or geopackage. You can use the resulting file to
        display borehole locations in GIS for instance.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_file kwargs.
        """
        self.header.to_file(out_file, **kwargs)

    def to_geoparquet(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write header data to geoparquet. You can use the resulting file to display borehole locations in GIS for instance.
        Please note that Geoparquet is supported by GDAL >= 3.5. For Qgis this means QGis >= 3.26

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_parquet kwargs.
        """
        self.header.to_parquet(out_file, **kwargs)

    def to_ipf(self, out_file: Union[str, WindowsPath], **kwargs):
        # TODO write the pandas dataframes to IPF
        pass

    def to_vtm(
        self,
        out_file: Union[str, WindowsPath],
        data_columns: List[str],
        radius: float = 1,
        vertical_factor: float = 1.0,
        **kwargs,
    ):
        """
        Save objects to VTM (Multiblock file, an XML VTK file pointing to multiple other VTK files).
        For viewing boreholes/cpt's in e.g. ParaView or other VTK viewers

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to vtm file to be written
        data_columns : List[str]
            Labels of data columns to include for visualisation. Can be columns that contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m, by default 1
        vertical_factor : float, optional
            Factor to correct vertical scale. e.g. when layer boundaries are given in cm use 0.01 to convert to m, by default 1.0
            It is not recommended to use this for vertical exaggeration, use viewer functionality for that instead.
        **kwargs :
            pyvista.MultiBlock.save kwargs.
        """
        if not self.__vertical_reference == "NAP":
            raise NotImplementedError(
                "VTM export for vertical references other than NAP not implemented yet"
            )

        vtk_object = borehole_to_multiblock(
            self.data, data_columns, radius, vertical_factor, **kwargs
        )
        vtk_object.save(out_file, **kwargs)

    def to_geodataclass(self, out_file: Union[str, WindowsPath], **kwargs):
        # TODO write the pandas dataframes to geodataclass (used for Deltares GEO DataFusionTools)
        pass
