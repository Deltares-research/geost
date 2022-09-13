import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from dataclasses import dataclass, field
from typing import List, Union


class Base(object):
    def __post_init__(self):
        # Intercept __post_init__ call when using super() in child classes
        # This is because builtin 'object' is always last in the MRO, but doesn't have a __post_init__
        # All classes must therefore inherit from Base, such that the MRO becomes: child > parent(s) > Base > object
        pass


@dataclass
class PointDataCollection(Base):
    """
    Base class for a collection of boreholes and/or CPTs
    """

    __source: Union[str, WindowsPath]
    __table: pd.DataFrame

    def __post_init__(self):
        self.__selected: pd.DataFrame = None
        self.__entries = pd.DataFrame(
            [
                [ind[0], ind[1], ind[2], ind[3], ind[4]]
                for ind in self.table.index
                if ind[-1] == 0
            ],
            columns=self.table.index.names[:-1],
        )

    def __new__(cls, *args, **kwargs):
        if cls is PointDataCollection:
            raise TypeError(
                f"Cannot construct {cls.__name__} directly: construct class from its children instead"
            )
        else:
            return object.__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}:\nsource = {self.source.name}"

    @property
    def source(self):
        return self.__source

    @property
    def entries(self):
        return self.__entries

    @property
    def table(self):
        return self.__table

    @property
    def length(self):
        return len(self.entries)

    @property
    def selected(self):
        return self.__selected

    def sel(self, index: str or List[str]):
        self.__selected = self.table.loc[index]

    def append(self, other):
        if self.__class__ == other.__class__:
            self.__table = pd.concat([self.table, other.table])
            self.__entries = pd.concat([self.entries, other.entries])

    def to_parquet(self, out_file: Union[str, WindowsPath], selected=False, **kwargs):
        self.__table.to_parquet(out_file, **kwargs)

    def to_csv(self, out_file: Union[str, WindowsPath], selected=False, **kwargs):
        self.__table.to_csv(out_file, **kwargs)
