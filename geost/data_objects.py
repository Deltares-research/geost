from abc import ABC, abstractmethod

from geost.mixins import PandasExportMixin
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
    def get(self):  # Not really sure if necessary
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
    @property
    def df(self):
        return self._df

    @df.setter
    @validate_data
    def df(self, df):
        self._df = df

    def get(self):  # Not really sure if necessary
        raise NotImplementedError()

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


class DiscreteData(AbstractData, PandasExportMixin):
    @property
    def df(self):
        return self._df

    @df.setter
    @validate_data
    def df(self, df):
        self._df = df

    def get(self):  # Not really sure if necessary
        raise NotImplementedError()

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
    print(LayeredData())
    print(DiscreteData())
