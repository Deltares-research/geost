import pandas as pd
from pathlib import Path, WindowsPath
from dataclasses import dataclass

from pysst.base import PointDataCollection


@dataclass(repr=False)
class BoreholeCollection(PointDataCollection):
    """
    BoreholeCollection class
    """

    def __post_init__(self):
        super().__post_init__()
        self.__classification_system: str = "5104"

    @property
    def classification_system(self):
        return self.__classification_system
