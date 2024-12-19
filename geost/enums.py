from enum import IntEnum, StrEnum
from typing import Iterable, Literal


class UnitEnum(IntEnum):
    """
    Extension of IntEnum to be used for classified values of stratigraphic units and
    lithologies in models to allow easy selections.

    """

    @classmethod
    def select_units(cls, units: str | Iterable[str]):
        """
        Select units by name.

        Parameters
        ----------
        units : str | Iterable[str]
            Name as string or array_like object of strings with names to select.

        Returns
        -------
        List[enum]
            List of the selected units.

        """
        units = cls.cast_to_list(units)
        return [u for u in cls if u.name in units]

    @classmethod
    def select_values(cls, values: int | float | Iterable[int | float]):
        """
        Select units by value.

        Parameters
        ----------
        units : int | float | Iterable[int | float]
            Number or array_like object of numbers for units to select.

        Returns
        -------
        List[enum]
            List of the selected units.

        """
        values = cls.cast_to_list(values)
        return [n for n in cls if n.value in values]

    @classmethod
    def to_dict(cls, key: Literal["unit", "value"] = "value") -> dict:
        """
        Create a mapping dictionary from UnitEnum with "unit": "value" or "value": "unit"
        mapping.

        Parameters
        ----------
        key : Literal["unit", "value"], optional
            Determine what to use for "key" ("unit" or "value"). The default is "value".

        Returns
        -------
        dict

        """
        if key == "value":
            return {unit.value: unit.name for unit in cls}
        elif key == "unit":
            return {unit.name: unit.value for unit in cls}
        else:
            raise ValueError("Input for 'key' is wrong, use: 'unit' or 'value'")

    @staticmethod
    def cast_to_list(value):
        if isinstance(value, (int, float, str)):
            value = [value]
        return value


class VerticalReference(StrEnum):
    DATUM = "Local datum"
    DEPTH = "Depth"
    # Add vertical references
