import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from dataclasses import dataclass

from pysst.base import PointDataCollection
from pysst.validate import BoreholeSchema, CptSchema
from pysst.analysis import top_of_sand
from pysst.analysis.interpret_cpt import calc_ic, calc_lithology


@dataclass(repr=False)
class BoreholeCollection(PointDataCollection):
    """
    BoreholeCollection class.
    """

    def __post_init__(self):
        super().__post_init__()
        self.__classification_system: str = "5104"
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

        cover_layer["cover_thickness"] = cover_layer["cover_thickness"].fillna(
            np.abs(cover_layer["end"])
        )

        return cover_layer[["nr", "cover_thickness"]]


@dataclass(repr=False)
class CptCollection(PointDataCollection):
    """
    CptCollection class.
    """

    def __post_init__(self):
        super().__post_init__()
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
