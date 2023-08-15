from typing import Optional

import pandas as pd

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

        header (pd.DataFrame): Header used for construction. see
         :py:attr:`~pysst.base.PointDataCollection.header`
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vertical_reference: str = "NAP",
        header: Optional[pd.DataFrame] = None,
        header_col_names: Optional[list] = None,
    ):
        super().__init__(
            data, vertical_reference, header=header, header_col_names=header_col_names
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

        header (pd.DataFrame): Header used for construction. see
         :py:attr:`~pysst.base.PointDataCollection.header`
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vertical_reference: str = "NAP",
        header: Optional[pd.DataFrame] = None,
    ):
        super().__init__(data, vertical_reference, header=header)

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
