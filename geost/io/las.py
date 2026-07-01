from functools import cache
from typing import Iterable, Literal

import lasio
import numpy as np
import pandas as pd

import geost
from geost.utils.unit_conversion import calculate_factor

WELL_LOG_STANDARD_NAMES = {
    "curve": {
        "depth": ["DEPTH", "DEPT"],
        "gamma": ["GAMMA", "GAM(NAT)", "GR"],
        "caliper": [
            "CALIPER",
            "CALI",
        ],
        "resistivity": ["RESISTIVITY", "RES", "RT"],
        "speed": ["SPEED", "SP"],
    },
    "well": {
        "nr": ["WELL", "WELL:1", "WELL:2", "SITE_WELL_NAME", "HOLE_NAME"],
        "x": ["X", "XCOORD", "EASTING"],
        "y": ["Y", "YCOORD", "NORTHING"],
        "surface": ["EGL", "Z", "EDF", "GROUND_LEVEL"],
        "strt": ["STRT", "START"],
        "stop": ["STOP"],
        "step": ["STEP"],
    },
}

WELL_LOG_STANDARD_UNITS = {
    "well": {
        "surface": "m",
        "strt": "m",
        "stop": "m",
        "step": "m",
    },
    "curve": {
        "depth": "m",
        "gamma": "gapi",
        "caliper": "mm",
        "resistivity": "ohm.m",
        "speed": "m/min",
    },
}

NULL_POLICY = [("-999999.0", "-999.25", "-999999.25", "-999.0", "-9999")]


@cache
def _standard_names(category: Literal["curve", "well"]) -> dict:
    """Return a dictionary mapping well log curve aliases to their standard names."""
    rename_map = {
        alias: standard
        for standard, aliases in WELL_LOG_STANDARD_NAMES[category].items()
        for alias in aliases
    }
    rename_map.update({k: k for k in WELL_LOG_STANDARD_NAMES[category].keys()})
    return rename_map


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
        checked_mnemonics = set()

        # Standardize well header mnemonics and units
        for headeritem in las.well:
            if headeritem.mnemonic in _standard_names("well"):
                standard_mnemonic = _standard_names("well")[headeritem.mnemonic]
                if standard_mnemonic not in checked_mnemonics:
                    try:
                        headeritem.mnemonic = standard_mnemonic
                        checked_mnemonics.add(standard_mnemonic)
                        # Convert header item value to standard unit
                        if (
                            headeritem.unit != ""
                            and standard_mnemonic in WELL_LOG_STANDARD_UNITS["well"]
                        ):
                            standard_unit = WELL_LOG_STANDARD_UNITS["well"][
                                standard_mnemonic.lower()
                            ]
                            headeritem.value = float(
                                headeritem.value
                            ) * calculate_factor(headeritem.unit.lower(), standard_unit)
                            headeritem.unit = standard_unit
                    except Exception as e:
                        if errors == "raise":
                            raise e
                        elif errors == "warn":
                            print(
                                UserWarning(
                                    f"Warning: Could not standardize header '{headeritem.mnemonic}': {e}"
                                )
                            )

        # Standardize curve mnemonics and units
        for curve in las.curves:
            if curve.mnemonic in _standard_names("curve"):
                try:
                    curve.mnemonic = _standard_names("curve")[curve.mnemonic]
                    # Convert curve item data to standard unit
                    if (
                        curve.unit != ""
                        and curve.mnemonic in WELL_LOG_STANDARD_UNITS["curve"]
                    ):
                        standard_unit = WELL_LOG_STANDARD_UNITS["curve"][
                            curve.mnemonic.lower()
                        ]
                        curve.data = curve.data * calculate_factor(
                            curve.unit.lower(), standard_unit
                        )
                        curve.unit = standard_unit
                except Exception as e:
                    if errors == "raise":
                        raise e
                    elif errors == "warn":
                        print(
                            UserWarning(
                                f"Warning: Could not standardize curve '{curve.mnemonic}': {e}"
                            )
                        )

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
            {header_item.mnemonic: [str(header_item.value)] for header_item in las.well}
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
