import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.analysis import layers


@pytest.fixture
def layered_data():
    top = [
        0.0,
        0.5,
        0.7,
        1.1,
        1.2,
        1.8,
        0.0,
        0.3,
        0.4,
        0.5,
        0.7,
        0.8,
        1.4,
        1.6,
        1.8,
        2.0,
        2.05,
        2.1,
        0.0,
        0.2,
    ]
    bottom = [
        0.5,
        0.7,
        1.1,
        1.2,
        1.8,
        2.8,
        0.3,
        0.4,
        0.5,
        0.7,
        0.8,
        1.4,
        1.6,
        1.8,
        2.0,
        2.05,
        2.1,
        2.6,
        0.2,
        0.4,
    ]
    lith = [
        "K",
        "L",
        "Z",
        "L",
        "Z",
        "Z",
        "K",
        "L",
        "K",
        "Z",
        "L",
        "K",
        "L",
        "Z",
        "Z",
        "L",
        "Z",
        "Z",
        "K",
        "K",
    ]
    ics = [3, 2.6, 2, 2.6, 2, 2, 3, 2.6, 3, 2, 2.6, 3, 2.6, 2, 2, 2.6, 2, 2, 3, 3]
    return pd.DataFrame(
        {
            "nr": np.repeat(["a", "b", "c"], [6, 12, 2]),
            "surface": np.repeat([0.3, 0.4, 0.25], [6, 12, 2]),
            "top": top,
            "bottom": bottom,
            "lith": lith,
            "ic": ics,  # soil type index
        }
    )


@pytest.fixture
def discrete_data(layered_data):
    # Number of layers discrtetized to 5 cm in layered_data
    nlayers = [10, 4, 8, 2, 12, 20, 6, 2, 2, 4, 2, 12, 4, 4, 4, 1, 1, 10, 4, 4]

    discrete = layered_data.loc[layered_data.index.repeat(nlayers)].reset_index(
        drop=True
    )

    discrete.insert(2, "depth", 0.05)
    discrete["depth"] = discrete.groupby("nr")["depth"].cumsum()
    return discrete.drop(columns=["top", "bottom"])


@pytest.mark.parametrize(
    "data, value, min_thickness, min_fraction, expected_ids, expected_tops",
    [
        ("layered_data", "Z", None, None, ["a", "b"], [0.7, 0.5]),
        ("discrete_data", "Z", None, None, ["a", "b"], [0.7, 0.5]),
        ("layered_data", "Z", 0.5, None, ["a", "b"], [1.2, 2.05]),
        ("discrete_data", "Z", 0.5, None, ["a", "b"], [1.2, 2.05]),
        ("layered_data", "Z", 0.5, 0.8, ["a", "b"], [0.7, 1.6]),
        ("discrete_data", "Z", 0.5, 0.8, ["a", "b"], [0.7, 1.6]),
        ("layered_data", ["L", "Z"], 0.5, None, ["a", "b"], [0.5, 1.4]),
        ("discrete_data", ["L", "Z"], 0.5, None, ["a", "b"], [0.5, 1.4]),
        (
            "layered_data",
            slice(1.2, 2.3),
            0.5,
            0.8,
            ["a", "b"],
            [0.7, 1.6],
        ),  # slice for sand
        (
            "discrete_data",
            slice(1.2, 2.3),
            0.5,
            0.8,
            ["a", "b"],
            [0.7, 1.6],
        ),  # slice for sand
        ("layered_data", "Z", 1, 0.99, ["a"], [1.2]),
        ("discrete_data", "Z", 1, 0.99, ["a"], [1.2]),
        ("layered_data", "Z", 10, None, None, None),  # Too large min_thickness
    ],
    ids=[
        "layered_defaults",
        "discrete_defaults",
        "layered_min_thickness",
        "discrete_min_thickness",
        "layered_both_criteria",
        "discrete_both_criteria",
        "layered_value_list",
        "discrete_value_list",
        "layered_slice",
        "discrete_slice",
        "layered_high_fraction",
        "discrete_high_fraction",
        "too_large_min_thickness",
    ],
)
def test_get_layer_top(
    data, value, min_thickness, min_fraction, expected_ids, expected_tops, request
):
    test_id = request.node.callspec.id
    column = "ic" if "slice" in test_id else "lith"

    top = layers.get_layer_top(
        request.getfixturevalue(data), column, value, min_thickness, min_fraction
    )
    if test_id == "too_large_min_thickness":
        assert top.empty
    else:
        assert isinstance(top, pd.DataFrame)
        assert_array_equal(top["nr"], expected_ids)
        assert_array_almost_equal(top["top"], expected_tops)


@pytest.mark.unittest
def test_get_layer_top_errors(layered_data):
    with pytest.raises(
        ValueError, match="Data must contain columns specifying depth intervals"
    ):
        layers.get_layer_top(layered_data.drop(columns=["top", "bottom"]), "lith", "Z")

    with pytest.raises(ValueError, match="'min_thickness' cannot be below zero."):
        layers.get_layer_top(layered_data, "lith", "Z", min_thickness=-0.1)

    with pytest.raises(ValueError, match="'min_fraction' must be between 0 and 1."):
        layers.get_layer_top(layered_data, "lith", "Z", min_fraction=-0.1)

    with pytest.raises(ValueError, match="'min_fraction' must be between 0 and 1."):
        layers.get_layer_top(layered_data, "lith", "Z", min_fraction=1.1)
