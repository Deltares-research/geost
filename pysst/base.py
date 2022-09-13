import pandas as pd
from pathlib import WindowsPath
from dataclasses import dataclass
from typing import List, Union


class Base(object):
    """
    Base class to intercept __post_init__ call when using super() in child classes
    This is because builtin 'object' is always last in the MRO, but doesn't have a __post_init__
    All classes must therefore inherit from Base, such that the MRO becomes: child > parent(s) > Base > object
    """

    def __post_init__(self):
        pass


@dataclass
class PointDataCollection(Base):
    """
    Dataclass for collections of pointdata, such as boreholes and CPTs

    Args:
        __table (pd.DataFrame): Dataframe containing borehole/CPT data

    """

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
        return f"{self.__class__.__name__}:\n# entries = {self.length}"

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

    def sel(self, index: Union[str, List[str]]):
        """
        Make a selection of the table based on borehole ids

        Parameters
        ----------
        index : Union[str, List[str]]
            Borehole id
        """
        self.__selected = self.table.loc[index]

    def append(self, other):
        """
        Append data of other object of the same type (e.g BoreholeCollection to BoreholeCollection)

        Parameters
        ----------
        other : Instance of the same type as self
            Another object of the same type, from which the data is appended to self
        """
        if self.__class__ == other.__class__:
            self.__table = pd.concat([self.table, other.table])
            self.__entries = pd.concat([self.entries, other.entries])

    def to_parquet(self, out_file: Union[str, WindowsPath], selected=False, **kwargs):
        """
        Write table to parquet file

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to parquet file to be written
        selected : bool, optional
            Use only selected data (True) or all data (False). Default False
        **kwargs
            pd.DataFrame.to_parquet kwargs
        """
        self.__table.to_parquet(out_file, **kwargs)

    def to_csv(self, out_file: Union[str, WindowsPath], selected=False, **kwargs):
        """
        Write table to csv file

        Parameters
        ----------
        out_file : Union[str, WindowsPath]
            Path to csv file to be written
        selected : bool, optional
            Use only selected data (True) or all data (False). Default False
        **kwargs
            pd.DataFrame.to_csv kwargs
        """
        self.__table.to_csv(out_file, **kwargs)
