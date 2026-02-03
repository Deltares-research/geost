import time
from pathlib import Path

import numpy as np
import pandas as pd

import geost
from geost import read_borehole_table
from geost.accessors.data import LayeredData
from geost.analysis.grainsize import calculate_grainsize_metrics
from geost.io import xml
from geost.utils import dataframe_to_geodataframe


def read_tno_grainsize_xlsx(filepath: str | Path) -> pd.DataFrame:
    """
    Read TNO grain size distribution data from an Excel file into a Pandas DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the TNO grain size distribution Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the grain size distribution data.
    """
    algemeen_data = pd.read_excel(
        filepath,
        sheet_name="monsters",
    )

    meet_data = pd.read_excel(
        filepath,
        sheet_name="meetwaarden",
    )
    meet_data = meet_data[meet_data["PARAMETER_NM"] == "KG-Klasse"]

    joined = meet_data.merge(
        algemeen_data,
        on="SAMPLE_NR",
        how="left",
    )

    joined = joined[
        [
            "NITG_NR",
            "SAMPLE_NR",
            "X_UTM31_WGS84_CRD",
            "Y_UTM31_WGS84_CRD",
            "QS.TOP_DEPTH/1000",
            "QS.BOTTOM_DEPTH/1000",
            "DESCRIPTION",
            "LOWER_LIMIT",
            "UPPER_LIMIT",
            "VALUE",
            "Voorbehandelings methode",
            "Bepalings methode",
        ]
    ]

    joined.rename(
        columns={
            "NITG_NR": "nr",
            "SAMPLE_NR": "sample_nr",
            "X_UTM31_WGS84_CRD": "x",
            "Y_UTM31_WGS84_CRD": "y",
            "QS.TOP_DEPTH/1000": "top",
            "QS.BOTTOM_DEPTH/1000": "bottom",
            "DESCRIPTION": "description",
            "LOWER_LIMIT": "d_low",
            "UPPER_LIMIT": "d_high",
            "VALUE": "percentage",
            "Voorbehandelings methode": "preparation_method",
            "Bepalings methode": "determination_method",
        },
        inplace=True,
    )

    return joined


def join_data(data_1: "LayeredData", data_2: "LayeredData") -> "LayeredData":
    """
    Join two LayeredData collections based on common identifiers.

    Parameters
    ----------
    data_1 : LayeredData
        The first LayeredData collection.
    data_2 : LayeredData
        The second LayeredData collection.

    Returns
    -------
    LayeredData
        A new LayeredData collection with joined data.
    """
    data_to_join = data_2.copy()
    data_to_join = data_to_join[data_to_join["nr"].isin(data_1["nr"].unique())]

    # columns_to_add = data_to_join.columns.difference(data_1.columns).tolist()

    if not data_to_join.empty:
        joined_data = pd.concat(
            [
                pd.merge_asof(
                    data_1[data_1["nr"] == bh].sort_values("depth"),
                    data_to_join[data_to_join["nr"] == bh].sort_values("depth"),
                    on="depth",
                    direction="backward",
                    suffixes=("", "_df2"),
                )
                for bh in set(data_1["nr"]).union(data_to_join["nr"])
            ]
        )

        return joined_data


if __name__ == "__main__":
    # # Read example data
    # filepath = r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\korrelgrootte_noordzee_20251002.xlsx"
    # grainsize_data = read_tno_grainsize_xlsx(filepath)

    # # Calculate d50 for each sample
    # d50_results = calculate_grainsize_metrics(
    #     grainsize_data, percentiles=[10, 50, 90], only_sand=True, fractions=True
    # )

    # # collection = d50_results.gstda
    # print(d50_results)

    # d50_results.to_parquet(
    #     r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\grainsize_metrics.parquet",
    #     index=False,
    # )

    bhrp_cores = geost.bro_api_read(
        "BHR-P", bbox=(141_000, 455_200, 142_500, 456_000), crs_epsg=28992
    )
    borehole_descriptions = geost.bro_api_read(
        "BHR-GT", bbox=(114_000, 607_200, 122_500, 613_000), crs_epsg=28992
    )
    sample_descriptions = geost.bro_api_read(
        "BHR-GT-samples", bbox=(114_000, 607_200, 122_500, 613_000), crs_epsg=28992
    )

    geost.config.validation.SKIP = True

    d50_results = pd.read_parquet(
        r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\grainsize_metrics.parquet"
    )
    d50_results_selection = dataframe_to_geodataframe(
        d50_results
    ).gsthd.select_within_polygons(
        r"n:\Projects\11212000\11212071\B. Measurements and calculations\geologie\vanRWS\MER_zoekgebieden_2028-2037_v8.shp",
        buffer=500,
    )
    d50_results_selection["depth"] = d50_results_selection["bottom"]

    # Dino boreholes
    boreholes = read_borehole_table(
        r"c:\Users\onselen\Lokale data\DINO extractie Noordzee\DINO_Extractie_20240815.parquet"
    )
    boreholes.change_horizontal_reference(32631)
    boreholes_selection = boreholes.select_within_polygons(
        r"n:\Projects\11212000\11212071\B. Measurements and calculations\geologie\vanRWS\MER_zoekgebieden_2028-2037_v8.shp",
        buffer=500,
    )

    # Extra WBO boreholes from RWS
    extra_boreholes = read_borehole_table(
        r"p:\430-tgg-data\DINO\Noordzeeboringen_niet_in_dino\RWS_WBO_northsea_cores_2022_2024.parquet",
        horizontal_reference=32631,
    )
    extra_boreholes_selection = extra_boreholes.select_within_polygons(
        r"n:\Projects\11212000\11212071\B. Measurements and calculations\geologie\vanRWS\MER_zoekgebieden_2028-2037_v8.shp",
        buffer=500,
    )

    # Joined collection
    full_collection = boreholes_selection.data.gstda.append_vertical_data(
        extra_boreholes_selection.data, overwrite_duplicates=True
    ).to_collection()

    # joined_data = join_data(d50_results_selection, full_collection.data)

    # Add surface and end data from full collection to d50_results if available
    merged = pd.merge(
        d50_results_selection, full_collection.header, on="nr", how="left"
    )
    no_match = merged["x_y"].isna()
    original_part = d50_results_selection.reset_index().iloc[no_match]
    matched_part = merged[~no_match].copy()
    matched_part["x"] = matched_part["x_y"]
    matched_part["y"] = matched_part["y_y"]
    matched_part["geometry"] = matched_part["geometry_y"]
    matched_part.drop(
        columns=[
            "x_y",
            "y_y",
            "geometry_y",
            "x_x",
            "y_x",
            "geometry_x",
        ],
        inplace=True,
    )
    original_part.set_index("index", inplace=True)
    original_part["surface"] = np.nan
    original_part["end"] = np.nan
    combined_d50_and_collection = pd.concat(
        [original_part, matched_part], ignore_index=True, axis=0
    ).sort_values(by=["nr", "depth"])

    d50_collection = combined_d50_and_collection.gstda.to_collection()

    d50_collection.to_geopackage(
        r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\grainsize_metrics_MER_selection.gpkg"
    )
    full_collection.to_geopackage(
        r"n:\Projects\11212000\11212071\B. Measurements and calculations\geologie\boreholes_MER_selection.gpkg"
    )

    print(d50_results_selection)
