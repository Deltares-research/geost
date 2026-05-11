import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.analysis.combine import (
    add_nearest_voxelmodel_variable,
    add_voxelmodel_variable,
)
from geost.base import Collection


@pytest.fixture
def strat_deeper_than_borehole():
    return pd.DataFrame(
        {
            "nr": ["A"] * 4,
            "strat": [1, 2, 3, 4],
            "bottom": [-1.5, -2.0, -4.3, -5.1],
        }
    )


@pytest.fixture
def strat_deeper_than_cpt():
    return pd.DataFrame(
        {
            "nr": ["a"] * 4,
            "strat": [1, 2, 3, 4],
            "bottom": [-1.5, -2.0, -7.5, -10],
        }
    )


@pytest.mark.unittest
def test_add_nearest_voxelmodel_variable_zero_tolerance(
    borehole_collection, voxelmodel
):
    result = add_nearest_voxelmodel_variable(
        borehole_collection,
        voxelmodel,
        ["strat", "lith"],
        tolerances=(0, 0, 0),
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (25, 9)
    assert_array_equal(
        result.data[["lith", "strat"]],
        np.full((25, 2), np.nan),
    )


@pytest.mark.unittest
def test_add_nearest_voxelmodel_variable_layered(borehole_collection, voxelmodel):
    result = add_nearest_voxelmodel_variable(
        borehole_collection,
        voxelmodel,
        ["strat", "lith"],
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (25, 9)
    assert_array_equal(
        result.data[["lith", "strat"]],
        np.array(
            [
                [np.nan, np.nan],
                [1.0, 1.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1.0, 1.0],
                [3.0, 2.0],
                [2.0, 2.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [2.0, 2.0],
                [2.0, 2.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [3.0, 1.0],
                [3.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [np.nan, np.nan],
            ]
        ),
    )


@pytest.mark.unittest
def test_add_nearest_voxelmodel_variable_discrete(cpt_collection, voxelmodel):
    result = add_nearest_voxelmodel_variable(
        cpt_collection,
        voxelmodel,
        ["strat", "lith"],
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (20, 11)
    assert_array_equal(
        result.data[["lith", "strat"]],
        np.array(
            [
                [np.nan, np.nan],
                [3.0, 1.0],
                [3.0, 1.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
    )


@pytest.mark.unittest
def test_add_voxelmodel_variable_layered(borehole_collection, voxelmodel):
    result = add_voxelmodel_variable(borehole_collection, voxelmodel, "strat")
    assert isinstance(result, Collection)
    assert result.data.shape == (35, 9)
    assert_array_equal(
        result.data["strat"],
        [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
            1.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            np.nan,
            np.nan,
        ],
    )
    assert_array_almost_equal(
        result.data["top"],
        [
            0.0,
            0.8,
            1.5,
            2.2,
            2.5,
            2.7,
            3.7,
            0.0,
            0.6,
            1.2,
            1.8,
            2.5,
            2.8,
            3.1,
            0.0,
            1.25,
            1.4,
            1.75,
            1.8,
            2.25,
            2.75,
            2.9,
            3.8,
            0.0,
            0.5,
            1.2,
            1.8,
            2.5,
            0.0,
            0.5,
            1.2,
            1.8,
            1.9,
            2.4,
            2.5,
        ],
    )
    assert_array_almost_equal(
        result.data["bottom"],
        [
            0.8,
            1.5,
            2.2,
            2.5,
            2.7,
            3.7,
            4.2,
            0.6,
            1.2,
            1.8,
            2.5,
            2.8,
            3.1,
            3.9,
            1.25,
            1.4,
            1.75,
            1.8,
            2.25,
            2.75,
            2.9,
            3.8,
            5.5,
            0.5,
            1.2,
            1.8,
            2.5,
            3.0,
            0.5,
            1.2,
            1.8,
            1.9,
            2.4,
            2.5,
            3.0,
        ],
    )


@pytest.mark.unittest
def test_add_voxelmodel_variable_discrete(cpt_collection, voxelmodel):
    result = add_voxelmodel_variable(cpt_collection, voxelmodel, "strat")
    assert isinstance(result, Collection)
    assert result.data.shape == (24, 10)
    assert_array_equal(
        result.data["strat"],
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    )
    assert_array_equal(
        result.data["depth"],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            4.1,
            4.6,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            2.8,
            3.0,
            3.3,
            3.5,
            4.0,
            4.5,
            5.0,
        ],
    )


@pytest.mark.unittest
def test_removes_if_column_is_present(borehole_collection, voxelmodel):
    borehole_collection.data["strat"] = 1000
    result = add_voxelmodel_variable(borehole_collection, voxelmodel, "strat")
    assert_array_equal(
        result.data["strat"],
        [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
            1.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            np.nan,
            np.nan,
        ],
    )
