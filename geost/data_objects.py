from abc import ABC, abstractmethod


class AbstractData(ABC):
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
    def to_parquet(self):
        # Can maybe already be concrete implementation (maybe from a Mixin) due to
        # simplicity of method
        pass

    @abstractmethod
    def to_csv(self):
        # Can maybe already be concrete implementation (maybe from a Mixin) due to
        # simplicity of method
        pass

    @abstractmethod
    def to_vtm(self):
        pass

    @abstractmethod
    def to_datafusiontools(self):
        pass


class LayeredData(AbstractData):
    pass


class DiscreteData(AbstractData):
    pass
