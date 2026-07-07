import re

import numpy as np
import pandas as pd

import geost


#TODO: testing
def ags_xyz_to_collection(file):
    """
    Reads an AGS XYZ file and returns a GeoST Collection.

    Parameters:
    file (str or file-like): The path to the AGS XYZ file or a file-like object.

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

    # Read data
    df = pd.read_csv(file, sep=r"\s+", skiprows=i - 1)
    column_names = df.columns[1:]
    df = df.iloc[:, :-1]
    df.columns = column_names
    df["nr"] = df["LINE_NO"].astype(str) + "_" + df["RECORD"].astype(str)

    # Restructure data to fit GeoST format
    df = df.loc[df.index.repeat(nlayers)].reset_index(drop=True)
    df["layer_nr"] = df.groupby("nr").cumcount() + 1

    data_columns = [
        col
        for col in column_names
        if re.match(r".*_\d+$", col) or re.match(r".*\d+$", col)
    ]
    data_column_base_names = set(
        ["_".join(col.split("_")[:-1]) for col in data_columns]
    )

    for col in data_column_base_names:
        if "STD" not in col:
            cols = [f"{col}_{i}" for i in range(1, nlayers + 1)]
            df[col] = df[cols].to_numpy()[np.arange(len(df)), df["layer_nr"] - 1]

    df.drop(columns=data_columns, inplace=True)

    collection = df.gst.to_collection(
        crs=crs,
        include_in_header=[
            col for col in df.columns if col not in data_column_base_names
        ],
    )

    return collection


if __name__ == "__main__":
    # Example usage
    file_path = r"c:\data\freshem\SCI_Smooth_Best_MOD_inv.xyz"
    df = ags_xyz_to_collection(file_path)
    print(df.head())
