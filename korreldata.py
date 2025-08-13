import pandas as pd

import geost
from geost import config
from geost.validation import DataSchemas, safe_validate

config.validation.VERBOSE = False
config.validation.DROP_INVALID = False

boreholes_all = geost.read_borehole_table(
    r"c:\Users\onselen\Lokale data\DINO extractie Noordzee\DINO_Extractie_20240815.parquet"
)
print(boreholes_all.n_points)

config.validation.DROP_INVALID = True

boreholes_filtered = geost.read_borehole_table(
    r"c:\Users\onselen\Lokale data\DINO extractie Noordzee\DINO_Extractie_20240815.parquet"
)
print(boreholes_filtered.n_points)

algemeen = pd.read_excel(
    r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\Korrelgrootte_Deltares_20250710.xlsx",
    sheet_name="algemeene gegevens",
)

meet = pd.read_excel(
    r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\Korrelgrootte_Deltares_20250710.xlsx",
    sheet_name="meetgegevens",
)
meet = meet[meet["PARAMETER_NM"] == "KG-Klasse"]

print(1)

joined = meet.merge(
    algemeen,
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
        "PARAMETER_NM",
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
        "PARAMETER_NM": "parameter_nm",
        "Voorbehandelings methode": "voorbehandelings_methode",
        "Bepalings methode": "bepalings_methode",
    },
    inplace=True,
)

joined = joined[~joined["nr"].isna()]
joined["d_low"][joined["d_low"].isna()] = 0
joined["d_high"][joined["d_high"].isna()] = 999999

config.validation.VERBOSE = True

joined = safe_validate(DataSchemas.grainsize_data, joined)

joined.to_parquet(
    r"n:\Projects\11212000\11212071\B. Measurements and calculations\korrelverdelingen\Korrelgrootte_Deltares_20250710_test.parquet"
)

print(1)
