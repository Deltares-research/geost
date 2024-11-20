from enum import IntEnum, StrEnum
from typing import Iterable


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

    @staticmethod
    def cast_to_list(value):
        if isinstance(value, (int, float, str)):
            value = [value]
        return value


class VerticalReference(StrEnum):
    DATUM = "Local datum"
    DEPTH = "Depth"
    # Add vertical references
