import operator
from functools import cache, lru_cache

UNITS = {
    # Length units, base = 1 mm
    "length": {
        "um": 0.001,
        "mm": 1,
        "cm": 10,
        "dm": 100,
        "m": 1000,
        "ohmm": 1000,  # idiots making well logs with ohm*m as unit for depth
        "pu": 1000,  # pu = 1 meter denk ik ofzo
        "f": 304.8,
        "ft": 304.8,
        "feet": 304.8,
        "in": 25.4,
        "hm": 100000,
        "km": 1000000,
    },
    # Area units, base = 1 mm^2
    "surface": {
        "mm2": 1,
        "mm^2": 1,
        "cm2": 100,
        "cm^2": 100,
        "dm2": 10000,
        "dm^2": 10000,
        "m2": 1000000,
        "m^2": 1000000,
        "f2": 92903.04,
        "f^2": 92903.04,
        "ft2": 92903.04,
        "ft^2": 92903.04,
        "feet2": 92903.04,
        "feet^2": 92903.04,
        "in2": 645.16,
        "in^2": 645.16,
        "ha": 100000000,
        "km2": 10000000000,
        "km^2": 10000000000,
    },
    # Volume units, base = 1 mm^3
    "volume": {
        "mm3": 1,
        "mm^3": 1,
        "cm3": 1000,
        "cm^3": 1000,
        "dm3": 1000000,
        "dm^3": 1000000,
        "m3": 1000000000,
        "m^3": 1000000000,
        "f3": 28316846.592,
        "f^3": 28316846.592,
        "ft3": 28316846.592,
        "ft^3": 28316846.592,
        "feet3": 28316846.592,
        "feet^3": 28316846.592,
        "in3": 16387.064,
        "in^3": 16387.064,
        "l": 1000000,
        "km3": 1000000000000000,
        "km^3": 1000000000000000,
    },
    # Time units, base = 1 s
    "time": {
        "s": 1,
        "min": 60,
        "h": 3600,
        "hour": 3600,
        "d": 86400,
        "day": 86400,
        "w": 604800,
        "week": 604800,
        "yr": 31536000,
        "year": 31536000,
    },
    # Gamma ray units, base = 1 API
    "gamma": {
        "api": 1,
        "gapi": 1,
        "api-gr": 1,
        "cps": 1,  # Assuming 1 API = 1 CPS for simplicity; this is not true and temporary
    },
    # Voltage units, base = 1 V
    "voltage": {
        "v": 1,
        "mv": 0.001,
        "kv": 1000,
    },
    # Electrical current units, base = 1 A
    "current": {
        "a": 1,
        "ma": 0.001,
        "ka": 1000,
    },
    # Electrical resistance units, base = 1 ohm
    "electrical_resistance": {
        "ohm": 1,
        "kohm": 1000,
    },
}

DIV_STRS = ("per", "/", "÷", "over")
MULT_STRS = ("times", "*", "x", "X", "·", ".", "•", "⨉", "⨯", "⨉", " ", "-")


@cache
def _all_units() -> dict[str, float]:
    """
    Return a dictionary of all units and their corresponding factors relative to the base
    unit of the corresponding category.
    """
    all_units = {}
    for _, units in UNITS.items():
        all_units.update(units)
    return all_units


@lru_cache(maxsize=128)
def parse_unit_expression(expr: str):
    """
    Parse a unit expression into its components.

    Parameters
    ----------
    expr : str
        The unit expression to parse.

    Returns
    -------
    dict
        A dictionary containing the left unit, right unit, and operator.

    Examples
    --------
    >>> parse_unit_expression("m/s")
    {'left': 'm', 'right': 's', 'operator': '/'}
    >>> parse_unit_expression("m per s")
    {'left': 'm', 'right': 's', 'operator': 'per'}
    >>> parse_unit_expression("ohm m")
    {'left': 'ohm', 'right': 'm', 'operator': ' '}
    """
    expr = expr.strip()

    for op in DIV_STRS + MULT_STRS:
        if op in expr:
            left, right = expr.split(op, 1)
            return {
                "left": left.strip(),
                "right": right.strip(),
                "operator": op,
            }

    return {"left": expr, "right": None, "operator": None}


@lru_cache(maxsize=128)
def calculate_factor(from_unit: str, to_unit: str) -> float:
    """
    Calculate the conversion factor between two unit expressions.

    Parameters
    ----------
    from_unit : str
        The unit expression to convert from.
    to_unit : str
        The unit expression to convert to.

    Returns
    -------
    float
        The conversion factor from the 'from_unit' to the 'to_unit'.

    Raises
    ------
    ValueError
        If the unit expressions are incompatible (one is division and the other is multiplication).

    Examples
    --------
    >>> calculate_factor("m", "cm")
    100.0
    >>> calculate_factor("m/s", "km/h")
    3.6
    >>> calculate_factor("ohm*m", "kohm*m")
    0.001
    """
    all_units = _all_units()
    from_unit = parse_unit_expression(from_unit)
    to_unit = parse_unit_expression(to_unit)

    if from_unit["operator"] is None and to_unit["operator"] is None:
        return all_units[from_unit["left"]] / all_units[to_unit["left"]]

    else:
        if from_unit["operator"] in DIV_STRS and to_unit["operator"] in DIV_STRS:
            op = operator.truediv
        elif from_unit["operator"] in MULT_STRS and to_unit["operator"] in MULT_STRS:
            op = operator.mul
        else:
            return 1
            raise ValueError(
                "Incompatible unit expressions. Both must be either division or multiplication."
            )

        return op(
            all_units[from_unit["left"]] / all_units[to_unit["left"]],
            all_units[from_unit["right"]] / all_units[to_unit["right"]],
        )
