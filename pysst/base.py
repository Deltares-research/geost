import pickle
from functools import reduce
from pathlib import WindowsPath
from typing import Iterable, List, Optional, TypeVar, Union

import pandas as pd

# Local imports
from pysst import spatial
from pysst.analysis import cumulative_thickness, layer_top
from pysst.export import borehole_to_multiblock, export_to_dftgeodata
from pysst.utils import MissingOptionalModule
from pysst.validate import fancy_warning
from pysst.validate.validation_schemes import (
    common_dataschema,
    common_dataschema_depth_reference,
    headerschema,
)

# Optional imports
try:
    import geopandas as gpd

    create_header = spatial.header_to_geopandas
except ModuleNotFoundError:
    gpd = MissingOptionalModule("geopandas")

    def create_header(x):
        return x


warn = fancy_warning(lambda warning_info: print(warning_info))

Coordinate = TypeVar("Coordinate", int, float)
GeoDataFrame = TypeVar("GeoDataFrame")

pd.set_option("mode.copy_on_write", True)


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
        header_col_names: Optional[list] = None,
    ):
        self.__vertical_reference = vertical_reference
        self.data = data
        if isinstance(header, pd.DataFrame):
            self.header = header
        else:
            self.reset_header(header_col_names)

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

    def reset_header(self, header_col_names=None):
        if isinstance(header_col_names, list):
            if not len(header_col_names) == 5:
                raise TypeError(
                    "The header aliases must be aliases for (in order): 'nr', 'x', 'y'",
                    "'mv', 'end'",
                )
        else:
            header_col_names = ["nr", "x", "y", "mv", "end"]

        header = self.data.drop_duplicates(subset=header_col_names[0])
        header = header[header_col_names].reset_index(drop=True)
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
        """
        This setter is called whenever the attribute 'header' is manipulated, either
        during construction (init) or when a user attempts to set the header on an
        instance of the object (e.g. instance.header = new_header_df). This will run the
        header through validation and warn the user of any potential problems.
        """
        headerschema.validate(header)
        if any(~header["nr"].isin(self.data["nr"].unique())):
            warn("Header does not cover all unique objects in data")
        self._header = header

    @data.setter
    def data(self, data):
        """
        This setter is called whenever the attribute 'data' is manipulated, either
        during construction (init) or when a user attempts to set the data on an
        instance of the object (e.g. instance.data = new_data_df). This will run the
        data through validation and warn the user of any potential problems.
        """
        if self.vertical_reference == "depth":
            common_dataschema_depth_reference.validate(data)
        else:
            common_dataschema.validate(data)
        self._data = data

    def add_columns_to_header(self, columns: Union[List, str], position="first"):
        """
        Add a column from the data table to the header table. E.g. if you want to add
        another borehole identification from the data table to be included in the
        header and geometry exports.

        Parameters
        ----------
        columns : Union[List, str]
            Name(s) of column(s) to add/. Can be a single string or a list of strings
        position : str, optional
            Which value to take for the header: "first" or "last", by default "first"
        """
        positions = {"first": 0, "last": -1}

        if isinstance(columns, str):
            columns = [columns]

        if not all([c in self.data.columns for c in columns]):
            raise IndexError(
                "One or more of the columns were not found in the data",
                " and cannot be added to the header",
            )

        header_copy = self.header.copy()
        for column in columns:
            header_copy[column] = self.data[column].iloc[positions[position]]

        self._header = header_copy

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
        selected_header = selected_header[~selected_header.duplicated()]
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
        selected_header = selected_header[~selected_header.duplicated()]
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
        selected_header = selected_header[~selected_header.duplicated()]
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
        selected_header = selected_header[~selected_header.duplicated()]
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
            Name of column that contains categorical data to use when looking for values
        selection_values : Union[str, Iterable]
            Values to look for in the column
        how : str
            Either "and" or "or". "and" requires all selction values to be present in
            column for selection. "or" will select the core if any one of the
            selection_values are found in the column. Default is "and".

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
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
            header=selected_header,
        )

    def select_by_depth(
        self,
        top_min: float = None,
        top_max: float = None,
        end_min: float = None,
        end_max: float = None,
        slice: bool = False,
    ):
        """
        Select data from depth constraints. If a keyword argument is not given it will
        not be considered. e.g. if you need only boreholes that go deeper than -500 m
        use only end_max = -500.

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

        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def select_by_length(self, min_length: float = None, max_length: float = None):
        """
        Select data from length constraints: e.g. all boreholes between 50 and 150 m
        long. If a keyword argument is not given it will not be considered.

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

        selected_header = selected_header[~selected_header.duplicated()]
        selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(
            selection,
            vertical_reference=self.vertical_reference,
            header=selected_header,
        )

    def slice_depth_interval(
        self,
        upper_boundary: Union[float, int] = None,
        lower_boundary: Union[float, int] = None,
        vertical_reference: str = "NAP",
    ):
        """
        Slice boreholes/cpts based on given upper and lower boundaries. e.g. if you want
        to cut off all layers below -10 m NAP and all layers above 2 m NAP then,
        provided that the vertical_reference is already "NAP", you can use:

        self.slice_vertical(lower_boundary=-10, upper_boundary=2)

        this returns an instance of the BoreholeCollection or CptCollection with only
        the sliced layers.

        Note #1: This method currently only slices along existing layer boundaries,
        which especially for boreholes could mean that thick layers may continue beyond
        the given boundaries.

        Note #2: The instance that is returned may contain a smaller number of objects
        if the slicing led to a removal of all layers of an object.

        Parameters
        ----------
        upper_boundary : Union[float, int], optional
            Every layer that starts above this is removed, by default 9999
        lower_boundary : Union[float, int], optional
            Every layer that starts below this is removed, by default -9999
        vertical_reference : str
            The vertical reference used in slicing. Either "NAP", "surface" or "depth"
            See documentation of the change_vertical_reference method for details
            on the possible vertical references. By default "NAP"

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        original_vertical_reference = self.vertical_reference
        self.change_vertical_reference(vertical_reference)

        data_sliced = self.data.copy()

        if vertical_reference != "depth":
            data_sliced = data_sliced[data_sliced["top"] > (lower_boundary or -9999)]
            data_sliced = data_sliced[data_sliced["bottom"] < (upper_boundary or 9999)]
        elif vertical_reference == "depth":
            data_sliced = data_sliced[data_sliced["top"] < (lower_boundary or 9999)]
            data_sliced = data_sliced[data_sliced["bottom"] > (upper_boundary or 1)]

        header_sliced = self.header.loc[
            self.header["nr"].isin(data_sliced["nr"].unique())
        ]

        result = self.__class__(
            data_sliced,
            vertical_reference=vertical_reference,
            header=header_sliced,
        )

        result.change_vertical_reference(original_vertical_reference)

        return result

    def slice_by_values(self):
        pass

    def get_area_labels(
        self, polygon_gdf: GeoDataFrame, column_name: str, include_in_header=False
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

        Returns
        -------
        pd.DataFrame
            Borehole ids and the polygon label they are in.
        """
        area_labels = spatial.find_area_labels(self.header, polygon_gdf, column_name)

        if include_in_header:
            self._header = self.header.merge(area_labels, on="nr")
        else:
            return area_labels

    def get_cumulative_layer_thickness(
        self, column: str, values: Union[str, List[str]], include_in_header=False
    ):
        """
        Get the cumulative thickness of layers of a certain type.

        For example, to get the cumulative thickness of the layers with lithology "K" in
        the column "lith" use:

        self.get_cumulative_layer_thickness("lith", "K")

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        values : str or List[str]
            Value(s) of entries in column that you want to find the cumulative thickness
            of.
        include_in_header :
            Whether to add the acquired data to the header table or not,
            By default False.
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
        self, column: str, values: Union[str, List[str]], include_in_header=False
    ):
        """
        Find the depth at which a specified layer first occurs.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        value : str
            Value of entries in column that you want to find top of.
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default
            False.
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

    def append(self, other):
        """
        Append data of other object of the same type (e.g BoreholeCollection to
        BoreholeCollection).

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
                f"{other.__class__}",
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
        Write header data to geoparquet. You can use the resulting file to display
        borehole locations in GIS for instance. Please note that Geoparquet is supported
        by GDAL >= 3.5. For Qgis this means QGis >= 3.26

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
        Save objects to VTM (Multiblock file, an XML VTK file pointing to multiple other
        VTK files). For viewing boreholes/cpt's in e.g. ParaView or other VTK viewers.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to vtm file to be written
        data_columns : List[str]
            Labels of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m, by default 1
        vertical_factor : float, optional
            Factor to correct vertical scale. e.g. when layer boundaries are given in cm
            use 0.01 to convert to m, by default 1.0. It is not recommended to use this
            for vertical exaggeration, use viewer functionality for that instead.
        **kwargs :
            pyvista.MultiBlock.save kwargs.
        """
        if not self.__vertical_reference == "NAP":
            raise NotImplementedError(
                'VTM export is not available for other vertical references than "NAP"'
            )

        vtk_object = borehole_to_multiblock(
            self.data, data_columns, radius, vertical_factor, **kwargs
        )
        vtk_object.save(out_file, **kwargs)

    def to_datafusiontools(
        self,
        columns: List[str],
        out_file: Union[str, WindowsPath] = None,
        encode: bool = True,
        **kwargs,
    ):
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
        out_file : Union[str, WindowsPath]
            Path to pickle file to be written.
        encode : bool
            Encode categorical data to additional binary columns (0 or 1).
            Also see explanation above. Default is True.
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
