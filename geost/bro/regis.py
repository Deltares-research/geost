from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xarray as xr

from geost.exceptions import MissingUnitError
from geost.utils.qgis import write_qml


@dataclass(repr=False)
class RegisUnits:
    df: pd.DataFrame
    _unit: str = "HYD_UNIT_CD"
    _desc: str = "DESCRIPTION"
    _seq: str = "SEQ_NR"
    _r: str = "RED_DEC"
    _g: str = "GREEN_DEC"
    _b: str = "BLUE_DEC"

    def __post_init__(self):
        if not {"strat", "lithok"}.issubset(self.df.columns):
            strat_lith = self._separate_strat_litho_regis()
            self.df = self.df.assign(**strat_lith)

        if self.df.index.name != self._unit:
            self.df.set_index(self._unit, inplace=True)

    def __repr__(self) -> str:  # pragma: no cover
        return self.df.__repr__()

    def _repr_html_(self) -> str:  # pragma: no cover
        """
        HTML representation for Jupyter notebook.

        """
        return self.df._repr_html_()

    def _separate_strat_litho_regis(self) -> pd.DataFrame:
        pattern = re.compile(
            r"""
            (?P<strat>^[A-Z-]+)  # Unit from beginning of string, capital letters and hyphens (e.g., NUKR, NUPZ-WA)
            (?P<lithok>[a-z]*)   # Zero or more lowercase letters (e.g., z, k, c), numbers are ignored
            """,
            re.VERBOSE,
        )

        # Replace Holocene and Gestuwd unit codes to ensure correct processing
        layer = self.df[self._unit].replace(
            {"NUhlc": "NUHLc", "NUgsc": "NUGSc"}, regex=True
        )
        result = layer.str.extract(pattern)
        result["strat"] = result["strat"].replace(
            {"NUHL": "NUhl", "NUGS": "NUgs"}, regex=True
        )  # Change Holocene and Gestuwd units back to original casing
        return result

    @property
    def unit(self) -> pd.Series:
        """
        `pandas.Series` containing the unique unit names in the metadata.

        """
        return self.df.index

    @property
    def description(self) -> pd.Series:
        """
        `pandas.Series` containing the unique descriptions in the metadata.

        """
        return self.df[self._desc]

    @property
    def colors_rgb(self) -> pd.DataFrame:
        """
        `pandas.DataFrame` containing the RGB color values for the units in the metadata.

        """
        return self.df[[self._r, self._g, self._b]]

    @property
    def regis_version(self) -> str:
        """
        REGIS II version number (e.g. v01r6) of the metadata.

        """
        return self.df.attrs.get("regis version", "unknown")

    def check_version_matches(self, regis: xr.Dataset) -> bool:
        """
        Check if the REGIS II metadata version matches the metadata version.

        Parameters
        ----------
        regis : xr.Dataset
            The REGIS II modeldata to check against.

        Returns
        -------
        bool
            True if the versions match, False otherwise.

        """
        if isinstance(regis, xr.DataArray):
            raise TypeError(
                "The model version cannot be found in a DataArray of a REGIS II variable, "
                "cannot check if the metadata version matches the model version. Please "
                "check the model version against the `xarray.Dataset` of REGIS II.",
            )

        title = regis.attrs.get("title", "unknown")
        version_pattern = re.compile(  # General pattern: v01r1..
            r"""
            v\d+  # Fixed letter 'v' followed by one or more digits
            r\d+  # Fixed letter 'r' followed by one or more digits
            (?:[a-zA-Z]\d+)* # Optional group for extra letter-digit combinations
            """,
            re.VERBOSE,
        )
        match = re.search(version_pattern, title)

        if match:
            model_version = match.group(0)
            version_matches = self.regis_version == model_version
        else:
            version_matches = False
            model_version = title

        if not version_matches:
            warnings.warn(
                f"REGIS II version mismatch: metadata version is {self.regis_version}, "
                f"model version is '{model_version}'",
                UserWarning,
            )

        return version_matches

    def to_qml(self, path: str, field: str | None = None) -> None:
        """
        Export the REGIS unit metadata as a QGIS categorized vector style.

        Parameters
        ----------
        path : str
            Path and filename where the generated QGIS style file will be written.
        field : str, optional
            Name of the field in the vector file containing REGIS unit codes.
            QGIS matches each feature's value to the metadata to choose its
            color. Defaults to ``self._unit``; category values always come
            from the REGIS metadata.

        Raises
        ------
        ValueError
            If the metadata table is empty or the target field name is invalid.

        """
        if self.df.empty:
            raise ValueError("Cannot export empty metadata.")
        if field is None:
            field = self._unit

        write_qml(
            path,
            self.df,
            value_column=self._unit,
            color_columns=(self._r, self._g, self._b),
            label_column=self._unit,
            description_column=self._desc,
            label_separator=" - ",
            field=field,
        )

    def select_unit(self, units: str | list[str] | xr.DataArray) -> RegisUnits:
        """
        Select metadata by unit names (e.g., 'NUhlc', 'NUBXz').

        Parameters
        ----------
        units : str | list[str] | xr.DataArray
            Unit name(s) to select. Can be a single value, a list of values, or an
            `xarray.Dataset` of REGIS II to check which units are in the area (see examples
            below).

        Returns
        -------
        :class:`~geost.bro.regis.RegisUnits`
            RegisUnits instance containing the metadata subset for the selected unit
            names.

        Raises
        ------
        MissingUnitError
            Raised if one or more of the specified unit names are not present in the
            metadata.

        Examples
        --------
        Select specific unit names from REGIS II:

        >>> regis_meta = geost.bro.regis_units()
        >>> selected_meta = regis_meta.select_unit("NUhlc") # Select a single unit name
        >>> selected_meta = regis_meta.select_unit(["NUhlc", "NUBXz"]) # Select multiple unit names

        Or use the the REGIS II model to select the corresponding metadata directly:

        >>> selected_meta = regis_meta.select_unit(regis)

        """
        if isinstance(units, (xr.DataArray, xr.Dataset)):
            units = units.gst.z
        elif isinstance(units, str):
            units = [units]

        try:
            sel = self.df.loc[units]
            return replace(self, df=sel)
        except KeyError:
            raise MissingUnitError(
                f"One or more unit names from {units} are not present in the metadata."
            )

    def _select_contains(
        self,
        substring: str | list[str],
        values: pd.Index | pd.Series,
        case_sensitive: bool,
    ) -> pd.DataFrame:
        """
        Helper method for `select_unit_contains` and `select_description_contains`.

        """
        if not isinstance(substring, str):
            substring = list(substring)
            substring = "|".join(substring)

        sel = self.df[values.str.contains(substring, case=case_sensitive)]

        if sel.empty:
            raise MissingUnitError(
                f"None of the selection substrings in {substring} are present in the metadata."
            )
        return sel

    def select_unit_contains(
        self, substring: str | list[str], case_sensitive: bool = False
    ) -> RegisUnits:
        """
        Select metadata by a substring for a partial unit code or codes.

        Parameters
        ----------
        substring : str | list[str]
            Substring or list of substrings to match within the unit codes.
        case_sensitive : bool, optional
            Whether the match should be case-sensitive, by default False.

        Returns
        -------
        class:`~geost.bro.regis.RegisUnits`
            RegisUnits instance containing the metadata subset for the selected units.

        Raises
        ------
        MissingUnitError
            Raised if none of the specified substrings are present in the metadata.

        Examples
        --------
        Select units from REGIS II that contain a specific substring:

        >>> regis_meta = geost.bro.regis_units()
        >>> selected_meta = regis_meta.select_unit_contains("NUBX") # Select a single substring
        >>> selected_meta = regis_meta.select_unit_contains(["NUBX", "NUOO"]) # Select multiple substrings

        """
        return replace(
            self, df=self._select_contains(substring, self.unit, case_sensitive)
        )

    def select_description_contains(
        self, substring: str | list[str], case_sensitive: bool = False
    ) -> RegisUnits:
        """
        Select metadata by a substring for a partial description or descriptions.

        Parameters
        ----------
        substring : str | list[str]
            Substring or list of substrings to match within the descriptions.
        case_sensitive : bool, optional
            Whether the match should be case-sensitive, by default False.

        Returns
        -------
        class:`~geost.bro.regis.RegisUnits`
            RegisUnits instance containing the metadata subset for the selected units.

        Raises
        ------
        MissingUnitError
            Raised if none of the specified substrings are present in the metadata.

        Examples
        --------
        Select units from REGIS II that contain a specific substring:

        >>> regis_meta = geost.bro.regis_units()
        >>> selected_meta = regis_meta.select_description_contains("Formatie van Oosterhout") # Select a single substring
        >>> selected_meta = regis_meta.select_description_contains(
        ...     ["Formatie van Oosterhout", "Formatie van Breda"]
        ... ) # Select multiple substrings

        """
        return replace(
            self, df=self._select_contains(substring, self.description, case_sensitive)
        )


def regis_units() -> RegisUnits:
    """
    Read the metadata of REGIS II for selection and translation of stratigraphic units in
    the model data. The metadata is loaded from an internally stored parquet file which is
    a copy of the metadata that is delivered with the REGIS II model when it is downloaded
    from the BRO (see: https://www.dinoloket.nl/modelbestanden-aanvragen).

    Returns
    -------
    :class:`~geost.bro.regis.RegisUnits`
        RegisUnits instance containing the stratigraphic metadata of REGIS II.

    """
    from geost.data import REGISTRY

    meta = pd.read_parquet(REGISTRY.fetch("regis_v02r2s3_metadata.parquet"))
    return RegisUnits(df=meta)
