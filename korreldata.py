import pandas as pd

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
        "Voorbehandelings methode",
        "Bepalings methode",
    ]
]
