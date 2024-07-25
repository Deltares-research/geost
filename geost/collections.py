from abc import ABC, abstractmethod

import geopandas as gpd
import pandas as pd

from geost.data_objects import AbstractData, DiscreteData, LayeredData
from geost.headers import AbstractHeader, LineHeader, PointHeader

type DataObject = DiscreteData | LayeredData
type HeaderObject = LineHeader | PointHeader


class AbstractCollection(ABC):
    @property
    @abstractmethod
    def header(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    def n_points(self):
        pass

    @property
    @abstractmethod
    def horizontal_reference(self):  # Move to header class in future refactor
        pass

    @property
    @abstractmethod
    def vertical_reference(self):  # move to data class in future refactor
        pass

    @header.setter
    @abstractmethod
    def header(self, header):
        pass

    @data.setter
    @abstractmethod
    def data(self, data):
        pass

    @horizontal_reference.setter
    @abstractmethod
    def horizontal_reference(self, to_epsg: int):
        pass

    @vertical_reference.setter
    @abstractmethod
    def vertical_reference(self, to_epsg: str):  # will use epsg after refactor
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def reset_header(self):
        pass

    @abstractmethod
    def check_header_to_data_alignment(self):
        pass

    @abstractmethod
    def check_and_coerce_crs(self):
        pass

    @abstractmethod
    def select_within_bbox(self):
        pass

    @abstractmethod
    def select_with_points(self):
        pass

    @abstractmethod
    def select_with_lines(self):
        pass

    @abstractmethod
    def select_within_polygons(self):
        pass

    @abstractmethod
    def select_by_depth(self):
        pass

    @abstractmethod
    def select_by_length(self):
        pass

    @abstractmethod
    def get_area_labels(self):
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
        # These kind of methods should go to a seperate layer_analysis module with
        # functions to cover such analyses
        pass

    @abstractmethod
    def get_layer_top(self):
        # These kind of methods should go to a seperate layer_analysis module with
        # functions to cover such analyses
        pass

    @abstractmethod
    def to_vtm(self):
        pass

    @abstractmethod
    def to_datafusiontools(self):
        # supporting this is low priority, perhaps even deprecate
        pass


class Collection(AbstractCollection):
    def __init__(
        self,
        header: HeaderObject,
        data: DataObject,
        horizontal_reference: int,
        vertical_reference: str,
    ):
        self.horizontal_reference = horizontal_reference
        self.vertical_reference = vertical_reference
        self.header = header
        self.data = data

    def __new__(cls, *args, **kwargs):
        if cls is Collection:
            raise TypeError(
                f"Cannot construct {cls.__name__} directly: construct class from its",
                "children instead",
            )
        else:
            return object.__new__(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n# header = {self.n_points}"

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        return self._data

    @property
    def n_points(self):  # No change
        """
        Number of objects in the collection.
        """
        return len(self.header.df)

    @property
    def horizontal_reference(self):  # Move to header class in future refactor
        return self._horizontal_reference

    @property
    def vertical_reference(self):  # move to data class in future refactor
        return self._vertical_reference

    @header.setter
    def header(self, header):
        if isinstance(header, LineHeader | PointHeader):
            self._header = header
        elif "_header" in self.__dict__.keys() and isinstance(header, gpd.GeoDataFrame):
            self._header = self._header.__class__(header)
        self.check_header_to_data_alignment()

    @data.setter
    def data(self, data):
        if isinstance(data, LayeredData | DiscreteData):
            self._data = data
        elif "_data" in self.__dict__.keys() and isinstance(data, pd.DataFrame):
            self._data = self._header.__class__(data)
        self.check_header_to_data_alignment()

    @horizontal_reference.setter
    def horizontal_reference(self, to_epsg):
        pass

    @vertical_reference.setter
    def vertical_reference(self, to_epsg):
        pass

    def get(self):
        raise NotImplementedError("Add function logic")

    def reset_header(self):
        raise NotImplementedError("Add function logic")

    def check_header_to_data_alignment(self):
        pass

    def check_and_coerce_crs(self):
        pass

    def select_within_bbox(self):
        raise NotImplementedError("Add function logic")

    def select_with_points(self):
        raise NotImplementedError("Add function logic")

    def select_with_lines(self):
        raise NotImplementedError("Add function logic")

    def select_within_polygons(self):
        raise NotImplementedError("Add function logic")

    def select_by_depth(self):
        raise NotImplementedError("Add function logic")

    def select_by_length(self):
        raise NotImplementedError("Add function logic")

    def get_area_labels(self):
        raise NotImplementedError("Add function logic")

    def select_by_values(self):
        raise NotImplementedError("Add function logic")

    def slice_depth_interval(self):
        raise NotImplementedError("Add function logic")

    def slice_by_values(self):
        raise NotImplementedError("Add function logic")

    def get_cumulative_layer_thickness(self):
        # Not sure if this should be here, potentially unsuitable with DiscreteData
        raise NotImplementedError("Add function logic")

    def get_layer_top(self):
        raise NotImplementedError("Add function logic")

    def to_vtm(self):
        raise NotImplementedError("Add function logic")

    def to_datafusiontools(self):
        # supporting this is low priority, perhaps even deprecate
        raise NotImplementedError("Add function logic")


class BoreholeCollection(Collection):
    pass


class CptCollection(Collection):
    pass


class LogCollection(Collection):
    pass
