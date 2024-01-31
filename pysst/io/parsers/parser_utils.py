from typing import NamedTuple, Union


class RDCoord(NamedTuple):
    x: Union[int, float]
    y: Union[int, float]
    epsg: int


class DDCoord(NamedTuple):
    lat: Union[int, float]
    lon: Union[int, float]
    epsg: int
