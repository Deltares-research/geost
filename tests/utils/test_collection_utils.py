import numpy as np
import pytest

import geost
from geost.base import Collection


@pytest.mark.unittest
def test_concat(borehole_collection):
    # Create a second collection with the same structure but different data and positional
    # column names
    header2 = borehole_collection.header.copy()
    header2["nr"] = ["F", "G", "H", "I", "J"]
    header2.rename(columns={"nr": "id", "x": "x-coord", "y": "y-coord"}, inplace=True)
    data2 = borehole_collection.data.copy()
    data2["nr"] = np.repeat(["F", "G", "H", "I", "J"], 5)
    data2["extra_col"] = np.random.rand(data2.shape[0])
    data2.rename(columns={"nr": "id", "x": "x-coord", "y": "y-coord"}, inplace=True)
    collection2 = Collection(
        data2, header=header2, vertical_datum=borehole_collection.vertical_datum
    )

    # Concatenate the two collections
    concatenated_outer = geost.concat([borehole_collection, collection2])
    concatenated_inner = geost.concat([borehole_collection, collection2], join="inner")

    # Check that the concatenated collection has the correct number of unique survey IDs
    assert set(concatenated_outer.header["nr"].unique()) == set(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    )
    assert set(concatenated_outer.data["nr"].unique()) == set(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    )

    # Check that the number of rows in the header and data is correct
    assert concatenated_outer.header.shape == (10, 5)
    assert concatenated_outer.data.shape == (50, 9)
    assert concatenated_inner.header.shape == (10, 5)
    assert concatenated_inner.data.shape == (50, 8)

    # Check that the positional columns were correctly aligned and renamed
    assert (
        concatenated_outer.header.gst.positional_columns
        == borehole_collection.header.gst.positional_columns
    )
    assert (
        concatenated_outer.data.gst.positional_columns
        == borehole_collection.data.gst.positional_columns
    )
