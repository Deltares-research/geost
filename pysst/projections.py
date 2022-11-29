import pyproj

# TODO check conversion accuracy. Ellipsoid or bro problem


def get_projection(epsg: str = "28992"):
    return pyproj.Proj(init="epsg:" + epsg)


def xy_to_ll(x, y, epsg):
    P = get_projection(epsg)
    return P(x, y, inverse=True)
