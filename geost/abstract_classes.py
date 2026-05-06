from abc import ABC, abstractmethod


class AbstractBase(ABC):  # pragma: no cover
    """
    Abstract base class describing methods that need to be defined in GeoST objects.
    """

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
    def select_by_elevation(self):
        pass

    @abstractmethod
    def select_by_length(self):
        pass

    @abstractmethod
    def spatial_join(self):
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
        pass

    @abstractmethod
    def get_layer_top(self):
        pass

    @abstractmethod
    def get_layer_base(self):
        pass

    @abstractmethod
    def to_pyvista_cylinders(self):
        pass

    @abstractmethod
    def to_pyvista_grid(self):
        pass
