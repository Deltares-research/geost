from pyproj import CRS, Transformer
from pyproj.crs import CompoundCRS


def horizontal_reference_transformer(
    epsg_from: str | int | CRS, epsg_to: str | int | CRS
):
    transformer = Transformer.from_crs(
        CRS(epsg_from), CRS(epsg_to), always_xy=True, accuracy=1.0
    )
    return transformer


def vertical_reference_transformer(
    horizontal_epsg: str | int | CRS,
    epsg_from: str | int | CRS,
    epsg_to: str | int | CRS,
):
    epsg_from = CompoundCRS("Temporary source CRS", [horizontal_epsg, epsg_from])
    epsg_to = CompoundCRS("Temporary target CRS", [horizontal_epsg, epsg_to])

    if epsg_from.is_vertical and epsg_to.is_vertical:
        transformer = Transformer.from_crs(epsg_from, epsg_to, always_xy=True)
    else:
        raise TypeError(
            "Given EPSG numbers must be known vertical datums, see spatialreference.org"
        )
    return transformer
