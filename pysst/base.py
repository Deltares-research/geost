import pandas as pd
import geopandas as gpd
from pathlib import WindowsPath
from dataclasses import dataclass
from typing import List, Union

from pysst import spatial
from pysst.write import borehole_to_vtk


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
        raise NotImplementedError
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

    def add_area_labels(
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

    def to_ipf(self, out_file: Union[str, WindowsPath], **kwargs):
        # TODO write the pandas dataframes to IPF
        pass

    def to_vtk(self, out_file: Union[str, WindowsPath], **kwargs):
        # TODO write the pandas dataframes to vtk
        vtk_object = borehole_to_vtk(self.data, self.header, **kwargs)

    def to_geodataclass(self, out_file: Union[str, WindowsPath], **kwargs):
        # TODO write the pandas dataframes to geodataclass
        pass
