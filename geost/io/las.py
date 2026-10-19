import re
from functools import cache
from typing import Literal

import lasio
import numpy as np
import pandas as pd

import geost

WELL_LOG_STANDARD_NAMES = {
    "Curves": {
        "depth": ["DEPTH", "DEPT"],
        "gamma": ["GAMMA", "GAM(NAT)", "GR"],
        "caliper": [
            "CALIPER",
            "CALI",
        ],
        "resistivity": ["RESISTIVITY", "RT"],
    },
    "Well": {
        "nr": ["WELL", "WELL:1", "WELL:2", "SITE_WELL_NAME", "HOLE_NAME"],
        "x": ["X", "XCOORD", "EASTING", "LOCX"],
        "y": ["Y", "YCOORD", "NORTHING", "LOCY"],
        "surface": ["EGL", "Z", "EDF", "GROUND_LEVEL"],
        "strt": ["STRT", "START"],
        "stop": ["STOP"],
        "step": ["STEP"],
    },
    "Parameter": {
        "x": ["X", "XCOORD", "EASTING", "LOCX"],
        "y": ["Y", "YCOORD", "NORTHING", "LOCY"],
        "casb": ["CASB", "CASING_BOREHOLE"],
    },
}

WELL_LOG_STANDARD_UNITS = {
    "Well": {
        "surface": "m",
        "strt": "m",
        "stop": "m",
        "step": "m",
    },
    "Curves": {
        "depth": "m",
        "gamma": "gapi",
        "gamma_raw": "cps",
        "caliper": "mm",
        "resistivity": "ohm.m",
    },
}

NULL_POLICY = [("-999999.0", "-999.25", "-999999.25", "-999.0", "-9999")]


@cache
def _standard_names(category: Literal["Well", "Curves", "Parameter"]) -> dict:
    """Return a dictionary mapping well log curve aliases to their standard names."""
    rename_map = {
        alias: standard
        for standard, aliases in WELL_LOG_STANDARD_NAMES[category].items()
        for alias in aliases
    }
    rename_map.update({k: k for k in WELL_LOG_STANDARD_NAMES[category].keys()})
    return rename_map


def _standardize_las_section(
    las: lasio.LASFile,
    section: Literal["Well", "Curves", "Parameter"],
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> None:
    """
    Standardize the mnemonics in a given section of a LAS file.

    Parameters
    ----------
    las : lasio.LASFile
        The LAS file to standardize.
    section : str
        The section to standardize ('Well', 'Curves', or 'Parameter').
    errors : {'raise', 'warn', 'ignore'}, optional
                How to handle errors during standardization. 'raise' will raise an exception,
        'warn' will print a warning, and 'ignore' will silently ignore errors. Default
        is 'raise'.

    Raises
    ------
    ValueError
        If the section is not 'Well', 'Curves', or 'Parameter'.
    """
    if section not in ["Well", "Curves", "Parameter"]:
        raise ValueError("Section must be 'Well', 'Curves', or 'Parameter'.")

    checked_mnemonics = set()

    for item in las.sections[section]:
        if item.mnemonic in _standard_names(section):
            try:
                if (
                    standard_name := _standard_names(section)[item.mnemonic]
                ) not in checked_mnemonics:
                    item.mnemonic = standard_name
                    checked_mnemonics.add(standard_name)
                standard_unit = WELL_LOG_STANDARD_UNITS.get(section, {}).get(
                    standard_name
                )
                if standard_unit and not getattr(item, "unit", None):
                    item.unit = standard_unit
            except Exception as e:
                if errors == "raise":
                    raise e
                elif errors == "warn":
                    print(
                        UserWarning(
                            f"Warning: Could not standardize {section} '{item.mnemonic}': {e}"
                        )
                    )
                elif errors == "ignore":
                    continue


def _normalize_header_value(value):
    if value is None or pd.isna(value) or value == "":
        return np.nan
    return value


def _normalize_curve_description(description):
    description = _normalize_header_value(description)
    if pd.isna(description):
        return description
    return re.sub(r"^\s*\d+\s+", "", str(description))


def _well_section_to_header(
    las: lasio.LASFile,
    header_columns: list[str],
    header_values: list[object],
) -> None:
    for header_item in las.well:
        mnemonic = header_item.mnemonic
        if not mnemonic:
            continue

        header_value = _normalize_header_value(header_item.value)
        header_unit = header_item.unit or np.nan
        if pd.isna(header_value) and pd.isna(header_unit):
            continue

        header_columns.append(mnemonic)
        header_values.append(header_value)
        if not pd.isna(header_unit):
            header_columns.append(f"{mnemonic}_unit")
            header_values.append(header_unit)


def _parameter_section_to_header(
    las: lasio.LASFile,
    header_columns: list[str],
    header_values: list[object],
) -> None:
    for header_item in las.params:
        mnemonic = header_item.mnemonic
        if not mnemonic:
            continue

        header_value = _normalize_header_value(header_item.value)
        if pd.isna(header_value):
            continue

        header_columns.append(mnemonic)
        header_values.append(header_value)
        header_unit = header_item.unit or np.nan
        if not pd.isna(header_unit):
            header_columns.append(f"{mnemonic}_unit")
            header_values.append(header_unit)


def _curve_section_to_header(
    las: lasio.LASFile,
    header_columns: list[str],
    header_values: list[object],
) -> None:
    for curve_item in las.curves:
        mnemonic = curve_item.mnemonic
        if not mnemonic:
            continue

        header_columns.append(mnemonic)
        header_values.append(
            _normalize_curve_description(getattr(curve_item, "descr", np.nan))
        )
        header_columns.append(f"{mnemonic}_unit")
        header_values.append(getattr(curve_item, "unit", np.nan) or np.nan)


def _las_to_header(las: lasio.LASFile) -> pd.DataFrame:
    header_columns: list[str] = []
    header_values: list[object] = []

    _well_section_to_header(las, header_columns, header_values)
    _parameter_section_to_header(las, header_columns, header_values)
    _curve_section_to_header(las, header_columns, header_values)

    return pd.DataFrame([header_values], columns=header_columns)


def standardize_well_log_las(
    las: lasio.LASFile,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> lasio.LASFile:
    """
    Standardize the curve names in a LAS file to their standard names.

    Parameters
    ----------
    las : lasio.LASFile
        The LAS file object to standardize.
    errors : {'raise', 'warn', 'ignore'}, optional
        How to handle errors during standardization. 'raise' will raise an exception,
        'warn' will print a warning, and 'ignore' will silently ignore errors. Default
        is 'raise'.

    Returns
    -------
    lasio.LASFile
        The LAS file with standardized curve names.
    """
    for section in ["Well", "Curves", "Parameter"]:
        _standardize_las_section(las, section, errors=errors)

    return las


def well_logs_to_collection(
    las: lasio.LASFile,
) -> geost.Collection:
    """
    Convert one LAS object representing a well log to a geost Collection.

    Parameters
    ----------
    las : lasio.LASFile
        The LAS file object to convert.
    Returns
    -------
    geost.Collection
        A geost Collection containing the data from the LAS files.
    """
    las_headers = []
    las_dataframes = []
    las_header = _las_to_header(las)
    las_dataframe = las.df().reset_index()

    try:
        pos_cols_header = las_header.gst.positional_columns
        if pos_cols_header["surface"] is None:
            las_header["surface"] = np.nan
            pos_cols_header["surface"] = "surface"
    except KeyError:
        las_header["nr"] = "unknown"
        pos_cols_header["nr"] = "nr"

    # Insert the positional columns from the header into the dataframe.
    for i, (col, value) in enumerate(pos_cols_header.items()):
        if value is not None and pos_cols_header[col] not in las_dataframe.columns:
            las_dataframe.insert(
                i, pos_cols_header[col], las_header[pos_cols_header[col]].values[0]
            )

    # Clean up header
    las_header = las_header.loc[:, ~las_header.columns.duplicated(keep="first")]

    # Add the header and dataframe to the lists for concatenation.
    las_headers.append(las_header)
    las_dataframes.append(las_dataframe)

    # Concatenate all LAS dataframes and headers into a single dataframe and header.
    data = pd.concat(las_dataframes, ignore_index=True, join="outer")
    header = pd.concat(las_headers, ignore_index=True, join="outer")
    collection = geost.Collection(data=data, header=header)

    return collection
