from geost.base import LineHeader, PointHeader


class HeaderFactory:
    def __init__(self):
        self._types = {}

    def register_header(self, type_, header):
        self._types[type_] = header

    @staticmethod
    def get_geometry_type(gdf):
        geometry_type = gdf.geom_type.unique()
        if len(geometry_type) == 0:
            return None
        elif len(geometry_type) > 1:
            return "Multiple geometries"
        else:
            return geometry_type[0]

    def create_from(self, gdf, **kwargs):
        geometry_type = self.get_geometry_type(gdf)
        header = self._types.get(geometry_type)
        if not header:
            raise ValueError(f"No Header type available for {geometry_type}")
        return header(gdf, **kwargs)


header_factory = HeaderFactory()
header_factory.register_header("Point", PointHeader)
header_factory.register_header("Line", LineHeader)
