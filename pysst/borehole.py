import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from dataclasses import dataclass

# Local imports
from pysst.base import PointDataCollection
#from pysst.validate import BoreholeSchema, CptSchema
from pysst.analysis import top_of_sand
from pysst.analysis.interpret_cpt import calc_ic, calc_lithology


class BoreholeCollection(PointDataCollection):
    """
    BoreholeCollection class.
    """

    def __init__(self, data: pd.DataFrame, vertical_reference: str = "NAP"):
        super().__init__(data, vertical_reference)
        self.__classification_system = "5104"
        # BoreholeSchema.validate(self.table, inplace=True)

    @property
    def classification_system(self):
        return self.__classification_system

    def cover_layer_thickness(self):
        """
        Return a DataFrame containing the borehole ids and corresponding cover
        layer thickness.

        """
        top_sand = pd.DataFrame(top_of_sand(self.data), columns=["nr", "top_sand"])

        cover_layer = top_sand.merge(self.header, on="nr", how="left")
        cover_layer["cover_thickness"] = cover_layer["mv"] - cover_layer["top_sand"]
        cover_layer_df = cover_layer[["nr", "cover_thickness"]]
        return cover_layer_df


class CptCollection(PointDataCollection):
    """
    CptCollection class.
    """

    def __init__(self, data: pd.DataFrame, vertical_reference: str = "NAP"):
        super().__init__(data, vertical_reference)
        # CptSchema.validate(self.table, inplace=True)

    def add_ic(self):
        """
        Calculate soil behaviour type index (Ic) for all CPT's in the collection
        """
        self.data["ic"] = calc_ic(self.data["qc"], self.data["friction_number"])

    def add_lithology(self):
        """
        Interpret lithoclass for all CPT's in the collection
        """
        if not "ic" in self.data.columns:
            self.add_ic()
        self.data["lith"] = calc_lithology(
            self.data["ic"], self.data["qc"], self.data["friction_number"]
        )
