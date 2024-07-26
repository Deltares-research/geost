import warnings
from abc import ABC, abstractmethod

import pandas as pd

from geost.headers import PointHeader
from geost.mixins import PandasExportMixin
from geost.utils import dataframe_to_geodataframe
from geost.validate.decorators import validate_data


class AbstractData(ABC):
    @property
    @abstractmethod
    def df(self):
        pass

    @df.setter
    @abstractmethod
    def df(self, df):
        pass

    @abstractmethod
    def to_header(self):
        pass

    @abstractmethod
    def to_collection(self):
        pass

    @abstractmethod
    def select_by_values(self):
        pass

    @abstractmethod
    def slice_depth_interval(self):
        pass

    @abstractmethod
    def slice_by_values(self):
        pass

    @abstractmethod
    def get_cumulative_layer_thickness(self):
        # Not sure if this should be here, potentially unsuitable with DiscreteData
        pass

    @abstractmethod
    def get_layer_top(self):
        pass

    @abstractmethod
    def to_vtm(self):
        pass

    @abstractmethod
    def to_datafusiontools(self):
        # supporting this is low priority, perhaps even deprecate
        pass


class LayeredData(AbstractData, PandasExportMixin):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __repr__(self):
        name = self.__class__.__name__
        data = self._df
        return f"{name} instance:\n{data}"

    @property
    def df(self):
        return self._df

    @df.setter
    @validate_data
    def df(self, df):
        self._df = df

    def to_header(self):
        header_columns = ["nr", "x", "y", "mv", "end"]
        header = self._df[header_columns].drop_duplicates().reset_index(drop=True)
        warnings.warn(
            (
                "Header does not contain a crs. Consider setting crs using "
                "header.set_horizontal_reference()."
            )
        )
        header = dataframe_to_geodataframe(header)
        return PointHeader(header)

    def to_collection(self):
        raise NotImplementedError()

    def select_by_values(self):
        raise NotImplementedError()

    def slice_depth_interval(self):
        raise NotImplementedError()

    def slice_by_values(self):
        raise NotImplementedError()

    def get_cumulative_layer_thickness(self):
        raise NotImplementedError()
        pass

    def get_layer_top(self):
        raise NotImplementedError()

    def to_vtm(self):
        raise NotImplementedError()

    def to_datafusiontools(self):
        raise NotImplementedError()


class DiscreteData(AbstractData, PandasExportMixin):
    def __init__(self, df):
        raise NotImplementedError(f"{self.__class__.__name__} not supported yet")

    @property
    def df(self):
        return self._df

    @df.setter
    @validate_data
    def df(self, df):
        self._df = df

    def to_header(self):
        raise NotImplementedError()

    def to_collection(self):
        raise NotImplementedError()

    def select_by_values(self):
        raise NotImplementedError()

    def slice_depth_interval(self):
        raise NotImplementedError()

    def slice_by_values(self):
        raise NotImplementedError()

    def get_cumulative_layer_thickness(self):
        raise NotImplementedError()
        pass

    def get_layer_top(self):
        raise NotImplementedError()

    def to_vtm(self):
        raise NotImplementedError()

    def to_datafusiontools(self):
        raise NotImplementedError()


if __name__ == "__main__":
    from pathlib import Path

    test_parquet = Path(__file__).parents[1] / r"tests/data/test_boreholes.parquet"
    boreholes = LayeredData(pd.read_parquet(test_parquet))
    boreholes.to_header()
