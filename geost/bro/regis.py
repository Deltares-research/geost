from __future__ import annotations

import re
from dataclasses import dataclass, replace

import pandas as pd


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

    def __repr__(self) -> str:
        return self.df.__repr__()

    def _separate_strat_litho_regis(self) -> pd.DataFrame:
        pattern = re.compile(
            r"""
            (?P<strat>^[A-Z-]+)  # Unit from beginning of string, capital letters and hyphens (e.g., NUKR, NUPZ-WA)
            (?P<lithok>[a-z]*)   # Zero or more lowercase letters (e.g., z, k, c), numbers are ignored
            """,
            re.VERBOSE,
        )
        layer = self.df[self._unit].str.replace(
            "NUhlc", "NUHLc"
        )  # Make sure Holocene is correctly processed
        result = layer.str.extract(pattern)
        result["strat"] = result["strat"].str.replace("NUHL", "NUhl")
        return result


def regis_units() -> RegisUnits:
    from geost.data import REGISTRY

    meta = pd.read_parquet(REGISTRY.fetch("regis_v02r2s3_metadata.parquet"))
    return RegisUnits(df=meta)
