from functools import lru_cache
from typing import Iterable, Literal

import lasio
import numpy as np
import pandas as pd

import geost

WELL_LOG_STANDARD_NAMES = {
    "curve": {
        "depth": ["DEPTH", "DEPT"],
        "gamma": ["GAMMA", "GAM(NAT)", "GR"],
        "caliper": [
            "CALIPER",
            "CALI",
        ],
        "resistivity": ["RESISTIVITY", "RES", "RT"],
    },
    "well": {
        "nr": ["WELL", "WELL:1", "WELL:2", "SITE_WELL_NAME"],
    },
}


@lru_cache(maxsize=2)
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
) -> lasio.LASFile | list[lasio.LASFile]:
    """
    Standardize the curve names in a LAS file to their standard names.

    Parameters
    ----------
    las_objects : lasio.LASFile | Iterable[lasio.LASFile]
        A LAS file object or an iterable of LAS file objects.

    Returns
    -------
    lasio.LASFile | list[lasio.LASFile]
        The LAS file(s) with standardized curve names.
    """
    if isinstance(las_objects, lasio.LASFile):
        las_objects = [las_objects]

    for las in las_objects:
        for headeritem in las.well:
            if headeritem.mnemonic in _standard_names("well"):
                headeritem.mnemonic = _standard_names("well")[headeritem.mnemonic]
        for curve in las.curves:
            if curve.mnemonic in _standard_names("curve"):
                curve.mnemonic = _standard_names("curve")[curve.mnemonic]

    # TODO: unit conversion for curves and well headers

    return las_objects[0] if len(las_objects) == 1 else las_objects


def well_logs_to_collection(las_objects: Iterable[lasio.LASFile]) -> geost.Collection:
    """
    Convert an iterable of LAS objects representing well logs to a geost Collection.

    Parameters
    ----------
    las_objects : Iterable[lasio.LASFile]
        An iterable of LAS file objects.

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

        # Standardize column names and rename columns based on aliases.
        las_header.gst.standardize_column_names()
        las_dataframe.gst.standardize_column_names()

        # Add the header and dataframe to the lists for concatenation.
        las_headers.append(las_header)
        las_dataframes.append(las_dataframe)

    # Concatenate all LAS dataframes and headers into a single dataframe and header.
    data = pd.concat(las_dataframes, ignore_index=True)
    header = pd.concat(las_headers, ignore_index=True)
    collection = geost.Collection(data=data, header=header)

    return collection
