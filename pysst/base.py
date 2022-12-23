import pandas as pd
import geopandas as gpd
from pathlib import WindowsPath
from dataclasses import dataclass
from typing import List, Union, TypeVar

from pysst import spatial
from pysst.export import borehole_to_multiblock

Coordinate = TypeVar("Coordinate", int, float)


class Base(object):
    """
    Base class to intercept __post_init__ call when using super() in child classes.
    This is because builtin 'object' is always last in the MRO, but doesn't have a __post_init__
    All classes must therefore inherit from Base, such that the MRO becomes: child > parent(s) > Base > object.
    """

    def __post_init__(self):
        pass


@dataclass
class PointDataCollection(Base):
    """
    Dataclass for collections of pointdata, such as boreholes and CPTs. The pysst module revolves around this class
    and includes all methods that apply generically to both borehole and CPT data, such as selection and export methods.

    Args:
        __data (pd.DataFrame): Dataframe containing borehole/CPT data.

    """

    __data: pd.DataFrame

    def __post_init__(self):
        self.__header = spatial.header_to_geopandas(
            self.data.drop_duplicates(subset=("nr"))[["nr", "x", "y", "mv", "end"]]
        ).reset_index(drop=True)

    def __new__(cls, *args, **kwargs):
        if cls is PointDataCollection:
            raise TypeError(
                f"Cannot construct {cls.__name__} directly: construct class from its children instead"
            )
        else:
            return object.__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n# header = {self.n_points}"

    @property
    def header(self):
        """
        This attribute is a dataframe of header (1 row per borehole/cpt) and includes:
        point id, x-coordinate, y-coordinate, surface level and end depth.
        """
        return self.__header

    @property
    def data(self):
        return self.__data

    @property
    def n_points(self):
        return len(self.header)

    def select_from_bbox(
        self,
        xmin: Coordinate,
        xmax: Coordinate,
        ymin: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):
        """
        Make a selection of the data based on a bounding box

        Parameters
        ----------
        xmin : Coordinate (float or int)
            Left x-coordinate of bbox
        xmax : Coordinate (float or int)
            Right x-coordinate of bbox
        ymin : Coordinate (float or int)
            Lower y-coordinate of bbox
        ymax : Coordinate (float or int)
            Upper y-coordinate of bbox
        invert: bool, default False
            Invert the selection

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = spatial.header_from_bbox(
            self.header, xmin, xmax, ymin, ymax, invert
        )
        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def select_from_points(
        self,
        point_gdf: gpd.GeoDataFrame,
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
        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def select_from_lines(
        self,
        line_gdf: gpd.GeoDataFrame,
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
        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def select_from_polygons(
        self,
        polygon_gdf: gpd.GeoDataFrame,
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
        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def select_from_present_values(self, select_dict: dict):
        """
        Select pointdata based on the presence of given values in the given columns. Can be used for example
        to return a BoreholeCollection of boreholes that contain peat in the lithology column. This can be achieved
        by passing e.g. the following argument to the method:

        {"lith": ["V"]},

        where the column "lith" contains lithologies and we will return any cores that have at least one time "V"
        in the lithology column. You look for multiple values as well, for example:

        {"lith": ["V", "Z"]}

        will return all boreholes that have either or both "V" and "Z" in the "lith" column. If you want to return
        boreholes that have both present at the same time you should pass the following argument:

        {
            "lith": ["V"],
            "lith": ["Z"],
        }

        Parameters
        ----------
        select_dict : dict
            Dict that contains the column names as key and a list of values to look for in this column

        Returns
        -------
        Child of PointDataCollection
            Instance of either BoreholeCollection or CptCollection.
        """
        selected_header = self.header.copy()
        for selection_key in select_dict.keys():
            notna = self.data["nr"][
                self.data[selection_key].isin(select_dict[selection_key])
            ].unique()
            selected_header = selected_header[selected_header["nr"].isin(notna)]

        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def select_from_depth(
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

        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def select_from_length(self, min_length: float = None, max_length: float = None):
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

        return self.__class__(
            self.data.loc[self.data["nr"].isin(selected_header["nr"])]
        )

    def get_area_labels(
        self, polygon_gdf: gpd.GeoDataFrame, column_name: str
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
        return spatial.find_area_labels(self.header, polygon_gdf, column_name)

    def append(self, other):
        """
        Append data of other object of the same type (e.g BoreholeCollection to BoreholeCollection).

        Parameters
        ----------
        other : Instance of the same type as self.
            Another object of the same type, from which the data is appended to self.
        """
        if self.__class__ == other.__class__:
            self.__data = pd.concat([self.data, other.data])
            self.__header = pd.concat([self.header, other.header])
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
        selected : bool, optional
            Use only selected data (True) or all data (False). Default False.
        **kwargs
            pd.DataFrame.to_parquet kwargs.
        """
        self.__data.to_parquet(out_file, **kwargs)

    def to_csv(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write data to csv file.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to csv file to be written.
        selected : bool, optional
            Use only selected data (True) or all data (False). Default False.
        **kwargs
            pd.DataFrame.to_csv kwargs.
        """
        self.__data.to_csv(out_file, **kwargs)

    def to_shape(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write header data to shapefile or geopackage. You can use the resulting file to display borehole locations in GIS for instance.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to shapefile to be written.
        selected : bool, optional
            Use only selected data (True) or all data (False). Default False.
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
        selected : bool, optional
            Use only selected data (True) or all data (False). Default False.
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
        vtk_object = borehole_to_multiblock(
            self.data, data_columns, radius, vertical_factor, **kwargs
        )
        vtk_object.save(out_file, **kwargs)

    def to_geodataclass(self, out_file: Union[str, WindowsPath], **kwargs):
        # TODO write the pandas dataframes to geodataclass (used for Deltares GEO DataFusionTools)
        pass
