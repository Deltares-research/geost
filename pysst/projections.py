import pyproj

# TODO check conversion accuracy. Ellipsoid or bro problem


def get_transformer(epsg_from, epsg_to):
    return pyproj.Transformer.from_crs(epsg_from, epsg_to, accuracy=1.0)


def xy_to_ll(x, y, epsg):
    T = get_transformer(epsg, 4326)
    transformed = list(T.itransform([(x, y)]))
    return transformed[0][1], transformed[0][0]
