import warnings

import pandas as pd
import pytest

from geost.validation import column_names as cn


@pytest.mark.unittest
def test_check_positional_columns():
    names = {
        k: list(v) for k, v in cn.POSSIBLE_COLUMN_NAMING.items()
    }  # Convert sets to lists because otherwise we can't index them

    longest_set = max(map(len, names.values()))

    for ii in range(longest_set):
        # Use every possible column name at least one time. Loop untill the longest
        # set of possible column names is exhausted. For each column type, we take
        # the ii-th name from the list of possible names, and if the list is shorter
        # than ii, we go back to the beginning of the list using modulo.
        nr = names["nr"][ii % len(names["nr"])]
        surface = names["surface"][ii % len(names["surface"])]
        end = names["end"][ii % len(names["end"])]
        x = names["x_coordinate"][ii % len(names["x_coordinate"])]
        y = names["y_coordinate"][ii % len(names["y_coordinate"])]
        top = names["top"][ii % len(names["top"])]
        bottom = names["depth"][ii % len(names["depth"])]

        df = pd.DataFrame(columns=[nr, surface, end, x, y, top, bottom])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cn.check_positional_column_presence(df)  # Should not raise an error

    # Test that a warning is raised when x/y columns are missing
    df = pd.DataFrame(
        columns=[
            "nr",
            "invalid_surface",
            "invalid_x",
            "invalid_y",
            "invalid_top",
            "invalid_bottom",
        ]
    )
    with pytest.warns() as record:
        cn.check_positional_column_presence(df)

    assert len(record) == 1
    message = str(record[0].message)
    assert "missing x/y coordinate columns" in message
    assert (
        "Use the 'column_mapper' argument to map your columns to 'x' and 'y'" in message
    )
    assert (
        "Input table is missing depth information on surface level and top and/or bottom depth"
        in message
    )
    assert (
        "Use the 'column_mapper' argument to map your columns to the expected depth columns"
        in message
    )

    # Test that a KeyError is raised when the survey ID column is missing
    df = pd.DataFrame(columns=["invalid_nr", "surface", "x", "y", "top", "bottom"])
    with pytest.raises(KeyError) as excinfo:
        cn.check_positional_column_presence(df)

    assert "missing a mandatory column that identifies survey IDs" in str(excinfo.value)
    assert (
        "Use the 'column_mapper' argument to specify which column identifies the survey ID"
        in str(excinfo.value)
    )
