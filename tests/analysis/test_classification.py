import numpy as np
import pandas as pd
import pytest

from geost.analysis.classification import nen5104_to_lithoclass


@pytest.mark.parametrize(
    "lith, zm, zmk, az, asilt, ak, lutum_pct, expected",
    [
        ("K", 100, "ZFC", "ZX", "SX", np.nan, 10, "Kleiig zand en zandige klei"),
        ("Z", 200, "ZMC", np.nan, np.nan, np.nan, 5, "Matig grof zand"),
        ("G", 2500, np.nan, np.nan, np.nan, np.nan, 0, "Grind"),
        ("SHE", 3000, np.nan, np.nan, np.nan, np.nan, 0, "Schelpen"),
        ("L", 50, np.nan, np.nan, np.nan, np.nan, 20, "Kleiig zand en zandige klei"),
        ("Z", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "Zand overig"),
        ("Z", np.nan, np.nan, np.nan, np.nan, "KX", np.nan, "Zand overig"),
        ("K", 150, "ZFC", "ZX", np.nan, np.nan, 15, "Kleiig zand en zandige klei"),
        ("Z", 300, np.nan, np.nan, np.nan, np.nan, 8, "Grof zand"),
        ("G", 1500, np.nan, np.nan, np.nan, np.nan, 0, "Grind"),
        ("SHE", 5000, np.nan, np.nan, np.nan, np.nan, 0, "Schelpen"),
        ("L", 75, np.nan, np.nan, np.nan, np.nan, 25, "Kleiig zand en zandige klei"),
        ("Z", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "Zand overig"),
        ("K", 120, np.nan, np.nan, "SX", np.nan, 12, "Kleiig zand en zandige klei"),
    ],
)
def test_nen5104_to_lithoclass(lith, zm, zmk, az, asilt, ak, lutum_pct, expected):
    assert (
        nen5104_to_lithoclass(
            pd.DataFrame(
                {
                    "lith": [lith],
                    "zm": [zm],
                    "zmk": [zmk],
                    "az": [az],
                    "asilt": [asilt],
                    "ak": [ak],
                    "lutum_pct": [lutum_pct],
                }
            ),
            names=True,
        ).iloc[0]
        == expected
    )


@pytest.mark.parametrize(
    "lith, az, asilt, ak, expected",
    [
        ("K", "ZX", "SX", np.nan, "Kleiig zand en zandige klei"),
        ("Z", np.nan, np.nan, np.nan, "Zand overig"),
        ("G", np.nan, np.nan, np.nan, "Grind"),
        ("SHE", np.nan, np.nan, np.nan, "Schelpen"),
        ("L", np.nan, np.nan, np.nan, "Kleiig zand en zandige klei"),
        ("Z", np.nan, np.nan, "KX", "Zand overig"),
        ("K", "ZX", np.nan, np.nan, "Kleiig zand en zandige klei"),
        ("V", np.nan, np.nan, np.nan, "Organische stof"),
    ],
)
def test_nen5104_to_lithoclass_no_optional_cols(lith, az, asilt, ak, expected):
    assert (
        nen5104_to_lithoclass(
            pd.DataFrame(
                {
                    "lith": [lith],
                    "az": [az],
                    "asilt": [asilt],
                    "ak": [ak],
                }
            ),
            names=True,
        ).iloc[0]
        == expected
    )


def test_nen5104_to_lithoclass_missing_required_cols():
    df = pd.DataFrame(
        {
            "lith": ["K"],
            "az": ["ZX"],
            # Missing 'asilt' and 'ak'
        }
    )
    with pytest.raises(
        ValueError, match="Missing required columns in DataFrame: asilt, ak"
    ):
        nen5104_to_lithoclass(df, names=True)
