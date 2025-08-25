from abc import ABC, abstractmethod


class AbstractHeader(ABC):  # pragma: no cover
    @property
    @abstractmethod
    def gdf(self):
        pass

    @property
    @abstractmethod
    def horizontal_reference(self):
        pass

    @property
    @abstractmethod
    def vertical_reference(self):
        pass

    @gdf.setter
    @abstractmethod
    def gdf(self, gdf):
        pass

    @abstractmethod
    def change_horizontal_reference(self):
        pass

    @abstractmethod
    def change_vertical_reference(self):
        pass

    @abstractmethod
    def get(self):
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


class AbstractData(ABC):  # pragma: no cover
    @property
    @abstractmethod
    def df(self):
        pass

    @property
    @abstractmethod
    def datatype(self):
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
    def select_by_condition(self):
        pass

    @abstractmethod
    def get_cumulative_thickness(self):
        # Not sure if this should be here, potentially unsuitable with DiscreteData
        pass

    @abstractmethod
    def get_layer_top(self):
        pass

    @abstractmethod
    def to_datafusiontools(self):
        pass


class AbstractCollection(ABC):  # pragma: no cover
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
    def select_by_condition(self):
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

    # @abstractmethod
    # def get_cumulative_thickness(self):
    #     # Not sure if this should be here, potentially unsuitable with DiscreteData
    #     # These kind of methods should go to a seperate layer_analysis module with
    #     # functions to cover such analyses
    #     pass

    # @abstractmethod
    # def get_layer_top(self):
    #     # These kind of methods should go to a seperate layer_analysis module with
    #     # functions to cover such analyses
    #     pass

    @abstractmethod
    def to_pyvista_cylinders(self):
        pass

    @abstractmethod
    def to_pyvista_grid(self):
        pass

    @abstractmethod
    def to_datafusiontools(self):
        pass
