import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from dataclasses import dataclass

from pysst.base import PointDataCollection
from pysst.validate import BoreholeSchema, CptSchema
from pysst.analysis import top_of_sand


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
        top_sand = pd.DataFrame(
            top_of_sand(self.data),
            columns=['nr', 'top_sand']
            )
        
        cover_layer = top_sand.merge(self.header, on='nr', how='left')
        cover_layer['cover_thickness'] = cover_layer['mv'] - cover_layer['top_sand']
        
        cover_layer['cover_thickness'] = cover_layer['cover_thickness'].fillna(
            np.abs(cover_layer['end'])
            )
        
        return cover_layer[['nr', 'cover_thickness']]


@dataclass(repr=False)
class CptCollection(PointDataCollection):
    """
    CptCollection class.
    """

    def __post_init__(self):
        super().__post_init__()
        CptSchema.validate(self.table, inplace=True)


if __name__ == "__main__":
    df = pd.read_csv(
        r'c:\Users\knaake\OneDrive - Stichting Deltares\Documents\deklaagdikte\dino_boringen_data.csv'
    )
    df = df.drop(columns=['Unnamed: 0'])
    
    data = BoreholeCollection(df)
    
    cover = data.cover_layer_thickness()
    
    print(cover)
