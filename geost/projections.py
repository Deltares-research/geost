import pyproj


def get_transformer(epsg_from, epsg_to):
    transformer = pyproj.Transformer.from_crs(epsg_from, epsg_to, accuracy=1.0)
    return transformer


def xy_to_ll(x, y, epsg):
    t = get_transformer(epsg, 4326)
    transformed = t.transform(x, y)
    return transformed
