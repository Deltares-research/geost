from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from geost.exceptions import MissingUnitError


class UnitType(Enum):
    STRAT = "strat"
    LITHOK = "lithok"


@dataclass(repr=False)
class GeotopUnits:
    """
    Container for stratigraphic and lithologic metadata information for the GeoTOP model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metadata information.
    metadata_type : UnitType
        Type of the metadata (stratigraphic or lithologic).
    data_var : str
        Name of the GeoTOP variable that corresponds to the metadata (e.g. "strat" for
        stratigraphic units, "lithok" for lithologic units).
    _nr : str, optional
        Column name for the voxel number in the metadata DataFrame. Default is "VOXEL_NR".
    _unit : str, optional
        Column name for the unit code in the metadata DataFrame. Default is "STR_UNIT_CD".
    _desc : str, optional
        Column name for the description in the metadata DataFrame. Default is "DESCRIPTION".
    _seq : str, optional
        Column name for the sequence number in the metadata DataFrame. Default is "SEQ_NR".
    _r : str, optional
        Column name for the red color value in the metadata DataFrame. Default is "RED_DEC".
    _g : str, optional
        Column name for the green color value in the metadata DataFrame. Default is "GREEN_DEC".
    _b : str, optional
        Column name for the blue color value in the metadata DataFrame. Default is "BLUE_DEC".
    """

    df: pd.DataFrame
    unit_type: UnitType
    data_var: str
    _nr: str = "VOXEL_NR"
    _unit: str = "STR_UNIT_CD"
    _desc: str = "DESCRIPTION"
    _seq: str = "SEQ_NR"
    _r: str = "RED_DEC"
    _g: str = "GREEN_DEC"
    _b: str = "BLUE_DEC"

    def __post_init__(self):
        if self.df.index.name != self._nr:
            self.df.set_index(self._nr, inplace=True)

    def __repr__(self) -> str:
        return f"{self.unit_type}\n{self.df.__repr__()}"

    @property
    def voxel_nr(self) -> pd.Index:
        """
        `pandas.Index` containing the unique voxel numbers in the metadata.

        """
        return self.df.index

    @property
    def unit(self) -> pd.Series:
        """
        `pandas.Series` containing the unique unit names in the metadata.

        """
        return self.df[self._unit]

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
    def geotop_version(self) -> str:
        """
        GeoTOP version number (e.g. v01r6) of the metadata.

        """
        return self.df.attrs.get("geotop version", "unknown")

    def check_version_matches(self, geotop: xr.Dataset) -> bool:
        """
        Check if the GeoTOP modeldata version matches the metadata version.

        Parameters
        ----------
        geotop : xr.Dataset
            The GeoTOP modeldata to check against.

        Returns
        -------
        bool
            True if the versions match, False otherwise.

        """
        if isinstance(geotop, xr.DataArray):
            raise TypeError(
                "The model version cannot be found in a DataArray of a GeoTOP variable, "
                "cannot check if the metadata version matches the model version. Please "
                "check the model version against the `xarray.Dataset` of GeoTOP.",
            )

        model_version = geotop.attrs.get("title", "unknown")

        version_matches = self.geotop_version in model_version

        if not version_matches:
            warnings.warn(
                f"GeoTOP version mismatch: metadata version is {self.geotop_version}, "
                f"model version is {model_version}",
                UserWarning,
            )

        return version_matches

    def select_voxel_nr(
        self, values: int | float | list[int | float] | xr.DataArray
    ) -> GeotopUnits:
        """
        Select metadata by voxel numbers (e.g. 1090, 2010).

        Parameters
        ----------
        values : int | float | list[int  |  float] | xr.DataArray
            Voxel number(s) to select. Can be a single value, a list of values, or an
            `xarray.DataArray` of a GeoTOP variable with voxelnumbers (see examples
            below).

        Returns
        -------
        :class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the selected voxel
            numbers.

        Raises
        ------
        MissingUnitError
            Raised if one or more of the specified voxel numbers are not present in the
            metadata.

        Examples
        --------
        Select specific voxel numbers from the stratigraphic metadata of GeoTOP:

        >>> gtp_meta = geost.bro.get_geotop_metadata_strat()
        >>> selected_meta = gtp_meta.select_voxel_nr(1130) # Select a single voxel number
        >>> selected_meta = gtp_meta.select_voxel_nr([1130, 2010]) # Select multiple voxel numbers

        Or use the stratigraphic variable from a GeoTOP model to select the corresponding
        metadata:

        >>> selected_meta = gtp_meta.select_voxel_nr(geotop["strat"])

        """
        if isinstance(values, xr.DataArray):
            values = np.unique(values)
            values = values[~np.isnan(values)]

        values = [values] if isinstance(values, (int, float)) else values

        try:
            sel = self.df.loc[values]
            return replace(self, df=sel)
        except KeyError:
            raise MissingUnitError(
                f"One or more voxel numbers from {values} are not present in the metadata."
            )

    def select_units(self, units: str | list[str]) -> GeotopUnits:
        """
        Select metadata by unit codes (e.g. "NUNIBA", "zm").

        Parameters
        ----------
        units : str | list[str]
            Unit code(s) to select. Can be a single value or a list of values (see examples
            below).

        Returns
        -------
        :class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the selected units.

        Raises
        ------
        MissingUnitError
            Raised if one or more of the specified units are not present in the metadata.

        Examples
        --------
        Select specific units from the stratigraphic metadata of GeoTOP:

        >>> gtp_meta_strat = geost.bro.get_geotop_metadata_strat()
        >>> selected_meta = gtp_meta_strat.select_units("NUNIBA") # Select a single unit
        >>> selected_meta = gtp_meta_strat.select_units(["NUNIBA", "NUNIHO"]) # Select multiple units

        Or select lithologies from the lithologic metadata of GeoTOP:

        >>> gtp_meta_lithok = geost.bro.get_geotop_metadata_lithok()
        >>> selected_meta = gtp_meta_lithok.select_units("kz") # Select a single lithology
        >>> selected_meta = gtp_meta_lithok.select_units(["zf", "zm", "zg"]) # Select multiple lithologies

        """
        if isinstance(units, str):
            sel = self.df[self.df[self._unit] == units]
        else:
            sel = self.df[self.df[self._unit].isin(units)]

        if sel.empty:
            raise MissingUnitError(
                f"None of the selection units in {units} are present in the metadata."
            )
        return replace(self, df=sel)

    def _select_contains(
        self, substring: str | list[str], column: str, case_sensitive: bool
    ) -> pd.DataFrame:
        """
        Helper method for `select_unit_contains` and `select_description_contains`.

        """
        if not isinstance(substring, str):
            substring = list(substring)
            substring = "|".join(substring)

        sel = self.df[self.df[column].str.contains(substring, case=case_sensitive)]

        if sel.empty:
            raise MissingUnitError(
                f"None of the selection substrings in {substring} are present in the metadata."
            )
        return sel

    def select_unit_contains(
        self, substring: str | list[str], case_sensitive: bool = False
    ) -> GeotopUnits:
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
        class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the selected units.

        Raises
        ------
        MissingUnitError
            Raised if none of the specified substrings are present in the metadata.

        Examples
        --------
        Select units from the stratigraphic metadata of GeoTOP that contain a specific substring:

        >>> gtp_meta_strat = geost.bro.get_geotop_metadata_strat()
        >>> selected_meta = gtp_meta_strat.select_unit_contains("NIHO") # Select a single substring
        >>> selected_meta = gtp_meta_strat.select_unit_contains(["NIHO", "NIBA"]) # Select multiple substrings

        """
        return replace(
            self, df=self._select_contains(substring, self._unit, case_sensitive)
        )

    def select_description_contains(
        self, substring: str | list[str], case_sensitive: bool = False
    ) -> GeotopUnits:
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
        class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the selected units.

        Raises
        ------
        MissingUnitError
            Raised if none of the specified substrings are present in the metadata.

        Examples
        --------
        Select units from the stratigraphic metadata of GeoTOP that contain a specific substring:

        >>> gtp_meta_strat = geost.bro.get_geotop_metadata_strat()
        >>> selected_meta = gtp_meta_strat.select_description_contains("Formatie van Naaldwijk") # Select a single substring
        >>> selected_meta = gtp_meta_strat.select_description_contains(
        ...     ["Formatie van Naaldwijk", "Formatie van Nieuwkoop"]
        ... ) # Select multiple substrings

        """
        return replace(
            self, df=self._select_contains(substring, self._desc, case_sensitive)
        )

    def get_antropogenic_units(self) -> GeotopUnits:
        """
        Select metadata for all anthropogenic units.

        Returns
        -------
        class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the anthropogenic
            units.

        """
        if self.unit_type == UnitType.LITHOK:
            raise ValueError(
                "Stratigraphic units are not present in the lithologic metadata of GeoTOP."
            )
        return replace(
            self,
            df=self._select_contains("antropoge", self._desc, case_sensitive=False),
        )

    def get_holocene_channel_units(self) -> GeotopUnits:
        """
        Select metadata for all Holocene channel units.

        Returns
        -------
        class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the Holocene
            channel units.

        """
        if self.unit_type == UnitType.LITHOK:
            raise ValueError(
                "Stratigraphic units are not present in the lithologic metadata of GeoTOP."
            )
        return replace(
            self,
            df=self._select_contains(
                "geulafzettingen", self._desc, case_sensitive=False
            ),
        )

    def get_holocene_units(self, include_channel_units: bool = True) -> GeotopUnits:
        """
        Select metadata for all Holocene units.

        Parameters
        ----------
        include_channel_units : bool, optional
            Whether to include Holocene channel units in the selection. The default is
            True.

        Returns
        -------
        class:`~geost.bro.geotop.GeotopUnits`
            GeotopUnits instance containing the metadata subset for the Holocene units.

        """
        if self.unit_type == UnitType.LITHOK:
            raise ValueError(
                "Stratigraphic units are not present in the lithologic metadata of GeoTOP."
            )
        undefined = 0
        antropogenic = [1000, 1005]
        older_units_begin = 3000

        holocene_mask = (self.voxel_nr < older_units_begin) & (
            ~self.voxel_nr.isin(antropogenic) & (self.voxel_nr != undefined)
        )
        holocene = self.df.loc[holocene_mask]

        if include_channel_units:
            channel_units = self.get_holocene_channel_units()
            holocene = pd.concat([holocene, channel_units.df])

        return replace(self, df=holocene)


def geotop_strat_units() -> GeotopUnits:
    """
    Read the stratigraphic metadata of GeoTOP for easy selection and translation of
    stratigraphic units in GeoTOP model data. The metadata is loaded from an internally
    stored parquet file which is a copy of the metadata that is delivered with the GeoTOP
    model when it is downloaded from the BRO (see: https://www.dinoloket.nl/modelbestanden-aanvragen).

    Returns
    -------
    :class:`~geost.bro.geotop.GeotopUnits`
        GeotopUnits instance containing the stratigraphic metadata of GeoTOP.

    """

    from geost.data import REGISTRY

    meta = pd.read_parquet(REGISTRY.fetch("geotop_v01r6s1_metadata_strat.parquet"))
    return GeotopUnits(meta, UnitType.STRAT, "strat")


def geotop_lithok_units() -> GeotopUnits:
    """
    Read the lithologic metadata of GeoTOP for easy selection and translation of
    lithologic units in GeoTOP model data. The metadata is loaded from an internally
    stored parquet file which is a copy of the metadata that is delivered with the GeoTOP
    model when it is downloaded from the BRO (see: https://www.dinoloket.nl/modelbestanden-aanvragen).

    Returns
    -------
    :class:`~geost.bro.geotop.GeotopUnits`
        GeotopUnits instance containing the lithologic metadata of GeoTOP.

    """
    from geost.data import REGISTRY

    meta = pd.read_parquet(REGISTRY.fetch("geotop_v01r6s1_metadata_lithok.parquet"))
    return GeotopUnits(meta, UnitType.LITHOK, "lithok", _unit="LITHO_CLASS_CD")


if __name__ == "__main__":
    gtp_meta_strat = geotop_strat_units()
    gtp_meta_lithok = geotop_lithok_units()

    print(gtp_meta_lithok)
