from abc import ABC, abstractmethod


class AbstractHeader(ABC):
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

    @abstractmethod
    def to_shape(self):
        # Can maybe already be concrete implementation (maybe from a Mixin) due to
        # simplicity of method
        pass

    @abstractmethod
    def to_geoparquet(self):
        # Can maybe already be concrete implementation (maybe from a Mixin) due to
        # simplicity of method
        pass


class PointHeader(AbstractHeader):
    def __init__(self, gdf):
        self.gdf = gdf

    def get(self):
        raise NotImplementedError('Add function logic')

    def select_within_bbox(self):
        raise NotImplementedError('Add function logic')

    def select_with_points(self):
        raise NotImplementedError('Add function logic')

    def select_with_lines(self):
        raise NotImplementedError('Add function logic')

    def select_within_polygons(self):
        raise NotImplementedError('Add function logic')

    def select_by_depth(self):
        raise NotImplementedError('Add function logic')

    def select_by_length(self):
        raise NotImplementedError('Add function logic')

    def get_area_labels(self):
        raise NotImplementedError('Add function logic')

    def to_shape(self):
        raise NotImplementedError('Add function logic')

    def to_geoparquet(self):
        raise NotImplementedError('Add function logic')


class LineHeader(AbstractHeader):
    def __init__(self, gdf):
        self.gdf = gdf

    def get(self):
        raise NotImplementedError('Add function logic')

    def select_within_bbox(self):
        raise NotImplementedError('Add function logic')

    def select_with_points(self):
        raise NotImplementedError('Add function logic')

    def select_with_lines(self):
        raise NotImplementedError('Add function logic')

    def select_within_polygons(self):
        raise NotImplementedError('Add function logic')

    def select_by_depth(self):
        raise NotImplementedError('Add function logic')

    def select_by_length(self):
        raise NotImplementedError('Add function logic')

    def get_area_labels(self):
        raise NotImplementedError('Add function logic')

    def to_shape(self):
        raise NotImplementedError('Add function logic')

    def to_geoparquet(self):
        raise NotImplementedError('Add function logic')


class HeaderFactory:
    def __init__(self):
        self._types = {}

    def register_header(self, type_, header):
        self._types[type_] = header

    @staticmethod
    def get_geometry_type(gdf):
        geometry_type = gdf.geom_type.unique()
        if len(geometry_type) > 1:
            return 'Multiple geometries'
        else:
            return geometry_type[0]

    def create_from(self, gdf, **kwargs):
        geometry_type = self.get_geometry_type(gdf)
        header = self._types.get(geometry_type)
        if not header:
            raise ValueError(f'No Header type available for {geometry_type}')
        return header(gdf, **kwargs)


header_factory = HeaderFactory()
header_factory.register_header('Point', PointHeader)
header_factory.register_header('Line', LineHeader)
