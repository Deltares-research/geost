import re
from typing import Literal

import numpy as np
import pandas as pd

import geost


# TODO: testing
def ags_xyz_to_collection(
    file,
    data_columns=["DEP_TOP", "DEP_BOT", "RHO", "RHO_STD"],
    trim_to_doi: bool = True,
    doi_type: Literal["standard", "conservative"] = "standard",
) -> geost.Collection:
    """
    Reads an AGS XYZ file and returns a GeoST Collection.

    Parameters:
    -----------
    file (str or file-like):
        The path to the AGS XYZ file or a file-like object.
    data_columns (list of str, optional):
        List of data columns to include in the collection. Default is ["DEP_TOP", "DEP_BOT", "RHO", "RHO_STD"].
    trim_to_doi (bool, optional):
        Whether to trim the data to the depth of investigation (DOI). Default is True.
    doi_type (str, optional):
        The type of depth of investigation (DOI) to use when trimming the data.
        Can be "standard" or "conservative". Default is "standard".

    Returns:
        geost.Collection: A GeoST Collection containing the data from the AGS XYZ file.
    """
    # Sniff the header to find the number of header lines and the CRS
    with open(file, "r") as f:
        header = ""
        for i, line in enumerate(f):
            if line.startswith("/"):
                header += line.lstrip("/").strip()
            else:
                break

    crs = re.search(r"epsg:(\d+)", header, re.IGNORECASE).group()
    nlayers = int(re.search(r"NUMBER OF LAYERS(\d+)", header).group(1))

    # Read data / skip column 0 that has no useful information
    df = pd.read_csv(file, sep=r"\s+", skiprows=i - 1)
    df = df.iloc[:, :-1].set_axis(df.columns[1:], axis="columns")
    df["nr"] = df["LINE_NO"].astype(str) + "_" + df["RECORD"].astype(str)

    # Restructure data to fit GeoST format
    df = df.loc[df.index.repeat(nlayers)].reset_index(drop=True)
    df["layer_nr"] = df.groupby("nr").cumcount() + 1

    data_columns_original = [
        col
        for col in df.columns
        if re.match(r".*_\d+$", col) or re.match(r".*\d+$", col)
    ]
    # data_column_base_names = set(
    #     ["_".join(col.split("_")[:-1]) for col in data_columns]
    # )

    for col in data_columns:
        # Try with underscore first, then without
        cols = [f"{col}_{i}" for i in range(1, nlayers + 1)]
        if not all(c in df.columns for c in cols):
            cols = [f"{col}{i}" for i in range(1, nlayers + 1)]
        df[col] = df[cols].to_numpy()[np.arange(len(df)), df["layer_nr"] - 1]

    df.drop(columns=data_columns_original, inplace=True)

    if trim_to_doi:
        if doi_type == "standard":
            df = df.loc[df["DOI_STANDARD"] >= df["DEP_BOT"]]
        elif doi_type == "conservative":
            df = df.loc[df["DOI_CONSERVATIVE"] >= df["DEP_BOT"]]
        # Set the last depth of each survey to the DOI that was used to trim
        df.loc[df.gst.last_row_survey, "DEP_BOT"] = (
            df["DOI_STANDARD"] if doi_type == "standard" else df["DOI_CONSERVATIVE"]
        )

    collection = df.gst.to_collection(
        crs=crs,
        include_in_header=[
            col for col in df.columns if col not in data_columns_original
        ],
    )

    return collection
