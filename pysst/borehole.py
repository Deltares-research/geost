from pathlib import Path, WindowsPath
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from shapely import LineString

# from pysst.validate import BoreholeSchema, CptSchema
from pysst.analysis import top_of_sand
from pysst.analysis.interpret_cpt import calc_ic, calc_lithology

# Local imports
from pysst.base import PointDataCollection


class BoreholeCollection(PointDataCollection):
    """
    Class for collections of borehole data.

    Users must use the reader functions in
    :py:mod:`~pysst.read` to create collections. The following readers generate Borehole
    objects:

    :func:`~pysst.read.read_sst_cores`, :func:`~pysst.read.read_nlog_cores`

    Args:
        data (pd.DataFrame): Dataframe containing borehole/CPT data.

        vertical_reference (str): Vertical reference, see
         :py:attr:`~pysst.base.PointDataCollection.vertical_reference`

        horizontal_reference (int): Horizontal reference, see
         :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`

        header (pd.DataFrame): Header used for construction. see
         :py:attr:`~pysst.base.PointDataCollection.header`
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vertical_reference: str = "NAP",
        horizontal_reference: int = 28992,
        header: Optional[pd.DataFrame] = None,
        is_inclined: bool = False,
    ):
        super().__init__(
            data,
            vertical_reference,
            horizontal_reference,
            header=header,
            is_inclined=is_inclined,
        )
        self.__classification_system = "5104"

    @property
    def classification_system(self):
        """
        Attribute: Borehole description protocol

        Returns
        -------
        str
            Borehole description protocol, e.g. "NEN5104"
        """
        return self.__classification_system

    def cover_layer_thickness(self, include_in_header=False):
        """
        Return a DataFrame containing the borehole ids and corresponding cover
        layer thickness.

        Parameters
        ----------
        include_in_header : bool, optional
            Whether to add the acquired data to the header table or not, by default ]
            False

        Returns
        -------
        pd.DataFrame
            Borehole ids and calculated cover layer thicknessess. If
            include_in_header = True, a column containing the generated data will be
            added inplace to :py:attr:`~pysst.base.PointDataCollection.header`.
        """
        top_sand = pd.DataFrame(top_of_sand(self.data), columns=["nr", "top_sand"])

        cover_layer = top_sand.merge(self.header, on="nr", how="left")
        cover_layer["cover_thickness"] = cover_layer["mv"] - cover_layer["top_sand"]
        cover_layer_df = cover_layer[["nr", "cover_thickness"]]

        if include_in_header:
            self.header = self.header.join(cover_layer_df["cover_thickness"])
        else:
            return cover_layer_df

    def to_qgis3d(self, out_file: Union[str, WindowsPath], **kwargs):
        """
        Write data to geopackage file that can be directly loaded in the Qgis2threejs
        plugin. Works only for layered (borehole) data.

        PLEASE NOTE:
        PySST does not support inclined boreholes yet. 3D visualisation may look a bit
        weird for the time being with layers being displaced instead of inclined.

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to geopackage file to be written.
        **kwargs
            geopandas.GeodataFrame.to_file kwargs. See relevant Geopandas documentation.

        """
        data_columns = [
            col
            for col in self.data.columns
            if col
            not in ["nr", "x", "y", "x_bot", "y_bot", "mv", "end", "top", "bottom"]
        ]

        data_to_write = dict(
            nr=self.data["nr"].values,
            top=self.data["top"].values.astype(float),
            bottom=self.data["bottom"].values.astype(float),
        )

        data_to_write.update(self.data[data_columns].to_dict(orient="list"))

        if self.is_inclined:
            geometries = [
                LineString([[x, y, top + 0.01], [x_bot, y_bot, bottom + 0.01]])
                for x, y, x_bot, y_bot, top, bottom in zip(
                    self.data["x"].values.astype(float),
                    self.data["y"].values.astype(float),
                    self.data["x_bot"].values.astype(float),
                    self.data["y_bot"].values.astype(float),
                    self.data["top"].values.astype(float),
                    self.data["bottom"].values.astype(float),
                )
            ]
        else:
            geometries = [
                LineString([[x, y, top + 0.01], [x, y, bottom + 0.01]])
                for x, y, top, bottom in zip(
                    self.data["x"].values.astype(float),
                    self.data["y"].values.astype(float),
                    self.data["top"].values.astype(float),
                    self.data["bottom"].values.astype(float),
                )
            ]

        geodataframe_result = gpd.GeoDataFrame(
            data=data_to_write, geometry=geometries, crs=self.horizontal_reference
        )
        geodataframe_result.to_file(Path(out_file), driver="GPKG")


class CptCollection(PointDataCollection):
    """
    Class for collections of CPT data.

    Users must use the reader functions in
    :py:mod:`~pysst.read` to create collections. The following readers generate CPT
    objects:

    :func:`~pysst.read.read_sst_cpts`, :func:`~pysst.read.read_gef_cpts`

    Args:
        data (pd.DataFrame): Dataframe containing borehole/CPT data.

        vertical_reference (str): Vertical reference, see
         :py:attr:`~pysst.base.PointDataCollection.vertical_reference`

        horizontal_reference (int): Horizontal reference, see
         :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`

        header (pd.DataFrame): Header used for construction. see
         :py:attr:`~pysst.base.PointDataCollection.header`
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vertical_reference: str = "NAP",
        horizontal_reference: int = 28992,
        header: Optional[pd.DataFrame] = None,
        is_inclined: bool = False,
    ):
        super().__init__(
            data,
            vertical_reference,
            horizontal_reference,
            header=header,
            is_inclined=is_inclined,
        )

    def add_ic(self):
        """
        Calculate soil behaviour type index (Ic) for all CPT's in the collection.

        The data is added to :py:attr:`~pysst.base.PointDataCollection.header`.
        """
        self.data["ic"] = calc_ic(self.data["qc"], self.data["friction_number"])

    def add_lithology(self):
        """
        Interpret lithoclass for all CPT's in the collection.

        The data is added to :py:attr:`~pysst.base.PointDataCollection.header`.
        """
        if "ic" not in self.data.columns:
            self.add_ic()
        self.data["lith"] = calc_lithology(
            self.data["ic"], self.data["qc"], self.data["friction_number"]
        )

    def as_boreholecollection(self):
        """
        Export CptCollection to BoreholeCollection. Requires the "lith" column to be
        present. Use the method :py:meth:`~pysst.borehole.CptCollection.add_lithology`

        Returns
        -------
        Instance of :class:`~pysst.borehole.BoreholeCollection`
        """
        if "lith" not in self.data.columns:
            raise IndexError(
                r"The column \"lith\" is required to convert to BoreholeCollection"
            )

        borehole_converted_dataframe = self.data[
            ["nr", "x", "y", "mv", "end", "top", "bottom", "lith"]
        ]
        cptcollection_as_bhcollection = BoreholeCollection(
            borehole_converted_dataframe,
            vertical_reference=self.vertical_reference,
            header=self.header,
        )
        return cptcollection_as_bhcollection
