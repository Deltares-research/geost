from typing import NamedTuple, Union


class RDCoord(NamedTuple):
    x: Union[int, float]
    y: Union[int, float]
    epsg: int


class DDCoord(NamedTuple):
    lat: Union[int, float]
    lon: Union[int, float]
    epsg: int


def safe_coerce(variable, dtype):
    try:
        variable = dtype(variable)
    except ValueError:
        Warning(f"Failed to coerce {variable} to {dtype}. Please check parsed data.")

    return variable
