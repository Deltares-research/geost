from functools import cache
from typing import Iterable, Literal

import lasio
import numpy as np
import pandas as pd

import geost
from geost.utils.unit_conversion import calculate_factor

WELL_LOG_STANDARD_NAMES = {
    "Curves": {
        "depth": ["DEPTH", "DEPT"],
        "gamma": ["GAMMA", "GAM(NAT)", "GR"],
        "caliper": [
            "CALIPER",
            "CALI",
        ],
        "resistivity": ["RESISTIVITY", "RES", "RT"],
        "speed": ["SPEED", "SP"],
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
        "speed": "m/min",
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
    Standardize the mnemonics and units in a given section of a LAS file.

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
                    # Convert units to standard units if applicable
                    if (
                        item.unit != ""
                        and standard_name in WELL_LOG_STANDARD_UNITS[section]
                    ):
                        standard_unit = WELL_LOG_STANDARD_UNITS[section][standard_name]
                        if section == "Well" or section == "Parameter":
                            # Convert single header value
                            item.value = float(item.value) * calculate_factor(
                                item.unit.lower(), standard_unit
                            )
                        elif section == "Curves":
                            # Convert the array of data values
                            item.data = item.data * calculate_factor(
                                item.unit.lower(), standard_unit
                            )
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


def standardize_well_log_las(
    las_objects: lasio.LASFile | Iterable[lasio.LASFile],
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> lasio.LASFile | list[lasio.LASFile]:
    """
    Standardize the curve names in a LAS file to their standard names.

    Parameters
    ----------
    las_objects : lasio.LASFile | Iterable[lasio.LASFile]
        A LAS file object or an iterable of LAS file objects.
    errors : {'raise', 'warn', 'ignore'}, optional
        How to handle errors during standardization. 'raise' will raise an exception,
        'warn' will print a warning, and 'ignore' will silently ignore errors. Default
        is 'raise'.

    Returns
    -------
    lasio.LASFile | list[lasio.LASFile]
        The LAS file(s) with standardized curve names.
    """
    if isinstance(las_objects, lasio.LASFile):
        las_objects = [las_objects]

    for las in las_objects:
        _standardize_las_section(las, "Well", errors=errors)
        _standardize_las_section(las, "Curves", errors=errors)
        _standardize_las_section(las, "Parameter", errors=errors)

    return las_objects[0] if len(las_objects) == 1 else las_objects


def well_logs_to_collection(
    las_objects: Iterable[lasio.LASFile],
    *,
    only_positional_columns: bool = True,
    join: Literal["inner", "outer"] = "outer",
) -> geost.Collection:
    """
    Convert an iterable of LAS objects representing well logs to a geost Collection.

    Parameters
    ----------
    las_objects : Iterable[lasio.LASFile]
        An iterable of LAS file objects.
    only_positional_columns : bool, optional
        If True, only include positional columns in the header of the Collection.
    join : {'inner', 'outer'}, optional
        How to join the header and data from the LAS files. 'inner' will keep only the
        columns that are present in all LAS files, while 'outer' will keep all columns,
        using NaN for missing values. Default is 'outer'.

    Returns
    -------
    geost.Collection
        A geost Collection containing the data from the LAS files.
    """
    las_headers = []
    las_dataframes = []
    for las in las_objects:
        las_header = pd.DataFrame(
            {
                header_item.mnemonic: [str(header_item.value)]
                for header_item in las.well + las.params
            }
        )
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
            if value is not None:
                las_dataframe.insert(
                    i, pos_cols_header[col], las_header[pos_cols_header[col]].values[0]
                )

        # Clean up header
        if only_positional_columns:
            las_header = las_header.loc[
                :, [p for p in pos_cols_header.values() if p is not None]
            ]
        las_header = las_header.loc[:, ~las_header.columns.duplicated(keep="first")]

        # Add the header and dataframe to the lists for concatenation.
        las_headers.append(las_header)
        las_dataframes.append(las_dataframe)

    # Concatenate all LAS dataframes and headers into a single dataframe and header.
    data = pd.concat(las_dataframes, ignore_index=True, join=join)
    header = pd.concat(las_headers, ignore_index=True, join=join)
    collection = geost.Collection(data=data, header=header)

    return collection
