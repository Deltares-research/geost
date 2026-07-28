import numpy as np
import pandas as pd

LITHOCLASS_TO_NAME = {
    -999: "NBE",
    1: "Organische stof",
    2: "Klei",
    3: "Kleiig zand en zandige klei",
    5: "Fijn zand",
    6: "Matig grof zand",
    7: "Grof zand",
    8: "Grind",
    9: "Schelpen",
    10: "Zand overig",
}

NEN5104_TO_LITHOCLASS_BASE = {
    "V": 1,
    "GY": 1,
    "DY": 1,
    "DET": 1,
    "HO": 1,
    "BRK": 1,
    "L": 3,
    "SHE": 9,
    "G": 8,
}


def _classify_clay(
    df: pd.DataFrame,
    col_names: dict,
) -> pd.Series:
    """
    Classify clay based on lith, az, asilt, ak and lutum_pct.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing mandatory columns
    col_names: dict
        Dictionary mapping the required column names to the actual column names in the DataFrame.

    Returns
    -------
    pd.Series
        Lithology classification.
    """
    # Boolean masks for clays, based on TNO `lithoklassificatie` function
    is_clay = df[col_names["lithology"]] == "K"
    is_sandy = (df[col_names["sand_admixture"]].isin(["ZX", "Z1", "Z2", "Z3"])) | (
        df[col_names["silt_admixture"]].isin(["SX", "S3", "S4"])
        | (df[col_names["lutum_pct"]] < 35)
    )
    # Clay, not specified by default -> 2 (Klei)
    df.loc[is_clay & ~is_sandy, "lithoklasse"] = 2

    # Clay, with sand or strong silt admixture -> 3 (Kleiig zand en zandige klei)
    df.loc[is_clay & is_sandy, "lithoklasse"] = 3

    return df


def _classify_sand(
    df: pd.DataFrame,
    col_names: dict,
) -> pd.Series:
    """
    Classify sand based on lith, zmk, az, asilt, ak and lutum_pct.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing mandatory columns
    col_names: dict
        Dictionary mapping the required column names to the actual column names in the DataFrame.

    Returns
    -------
    pd.Series
        Lithology classification.
    """
    # Boolean masks for sands, based on TNO `lithoklassificatie` function
    is_sand = df[col_names["lithology"]].isin(["Z", "GCZ"])
    is_clayey = (df[col_names["clay_admixture"]].isin(["K3", "KX"])) | (
        df[col_names["lutum_pct"]] >= 5
    )
    is_fine = (
        (df[col_names["sand_median"]] >= 63) & (df[col_names["sand_median"]] < 150)
    ) | (df[col_names["sand_median_class"]].isin(["ZFC", "ZUF", "ZUFO", "ZZF", "ZZFO"]))
    is_medium = (
        (df[col_names["sand_median"]] >= 150) & (df[col_names["sand_median"]] < 300)
    ) | (df[col_names["sand_median_class"]].isin(["ZMC", "ZMF", "ZMFO", "ZMG", "ZMGO"]))
    is_coarse = (
        (df[col_names["sand_median"]] >= 300) & (df[col_names["sand_median"]] < 2000)
    ) | (df[col_names["sand_median_class"]].isin(["ZGC", "ZZG", "ZZGO", "ZUG", "ZUGO"]))
    is_gravel = df[col_names["sand_median"]] >= 2000

    # Sand, not specified by default -> 10 (Zand overig)
    df.loc[is_sand, "lithoklasse"] = 10

    # Sand, with clay content -> 3 (Kleiig zand en zandige klei)
    df.loc[is_sand & is_clayey & is_fine, "lithoklasse"] = 3

    # Sand, no clay content, fine based on sand median or median class -> 5 (Fijn zand)
    df.loc[is_sand & is_fine & ~is_clayey, "lithoklasse"] = 5

    # Sand, medium based on sand median or median class -> 6 (Matig grof zand)
    df.loc[is_sand & is_medium, "lithoklasse"] = 6

    # Sand, coarse based on sand median or median class -> 7 (Grof zand)
    df.loc[is_sand & is_coarse, "lithoklasse"] = 7

    # Sand, very coarse so classify as gravel -> 8 (Grind)
    df.loc[is_sand & is_gravel, "lithoklasse"] = 8

    return df


def nen5104_to_lithoclass(
    df: pd.DataFrame,
    names: bool = False,
    lithology: str = "lith",
    sand_median_class: str = "zmk",
    sand_median: str = "zm",
    sand_admixture: str = "az",
    silt_admixture: str = "asilt",
    clay_admixture: str = "ak",
    lutum_pct: str = "lutum_pct",
) -> pd.Series:
    """
    Map NEN5104 descriptions to lithoclasses that are used in e.g. GeoTOP. It is based
    on the TNO `lithoklassificatie` function, but vectorized to be fast when applied to
    a DataFrame. The classification is based on columns for lithology, sand median class,
    sand median, sand admixture, silt admixture, clay admixture, and lutum percentage.

    - lithology, e.g. "Z", "K", "L", "G", "SHE", etc.
    - sand median class, e.g. "ZFC", "ZUF", "ZUFO", "ZZF", "ZZFO"
    - sand median in μm, e.g. 63, 150, 300, 2000
    - sand admixture, e.g. "ZX", "Z1", "Z2", "Z3"
    - silt admixture, e.g. "SX", "S3", "S1"
    - clay admixture, e.g. "K3", "KX"
    - lutum percentage, e.g. 5, 10, 20,

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the following required columns.
    names: bool, optional
        If True, return the lithoclass names instead of numeric codes.
    lithology: str, optional
        Column name for lithology. By default, "lith", in accordance to DINO database.
    sand_median_class: str, optional
        Column name for sand median class. By default, "zmk", in accordance to DINO database.
    sand_median: str, optional
        Column name for sand median. By default, "zm", in accordance to DINO database.
    sand_admixture: str, optional
        Column name for sand admixture. By default, "az", in accordance to DINO database.
    silt_admixture: str, optional
        Column name for silt admixture. By default, "asilt", in accordance to DINO database.
    clay_admixture: str, optional
        Column name for clay admixture. By default, "ak", in accordance to DINO database.
    lutum_pct: str, optional
        Column name for lutum percentage. By default, "lutum_pct", in accordance to DINO database.

    Returns
    -------
    pd.Series
        Lithology classification.
    """
    col_names = {
        "lithology": lithology,
        "sand_median_class": sand_median_class,
        "sand_median": sand_median,
        "sand_admixture": sand_admixture,
        "silt_admixture": silt_admixture,
        "clay_admixture": clay_admixture,
        "lutum_pct": lutum_pct,
    }
    # Check if required columns are present in the DataFrame
    required_cols = ["lithology", "sand_admixture", "silt_admixture", "clay_admixture"]
    optional_cols = ["sand_median_class", "sand_median", "lutum_pct"]
    missing_required_cols = [
        col_names[col] for col in required_cols if col_names[col] not in df.columns
    ]
    missing_optional_cols = [
        col_names[col] for col in optional_cols if col_names[col] not in df.columns
    ]

    if missing_required_cols:
        raise ValueError(
            f"Missing required columns in DataFrame: {', '.join(missing_required_cols)}"
        )
    elif missing_optional_cols:
        for col in missing_optional_cols:
            df[col] = np.nan

    # Classify lithology, initially default to -999 (NBE) for not classified
    df["lithoklasse"] = (
        df[col_names["lithology"]]
        .map(NEN5104_TO_LITHOCLASS_BASE)
        .fillna(-999)
        .astype(int)
    )
    df = _classify_clay(df, col_names)
    df = _classify_sand(df, col_names)

    if names:
        df.replace({"lithoklasse": LITHOCLASS_TO_NAME}, inplace=True)

    return df["lithoklasse"]
