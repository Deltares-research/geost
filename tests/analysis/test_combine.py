import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from geost.analysis.combine import (
    add_model_data,
    add_nearest_voxelmodel_variable,
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
        borehole_collection, voxelmodel, ["strat", "lith"], tolerances=(0, 0, 0)
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
        borehole_collection, voxelmodel, ["strat", "lith"]
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (25, 9)
    assert_array_equal(
        result.data[["lith", "strat"]],
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
        ],
    )


@pytest.mark.unittest
def test_add_nearest_voxelmodel_variable_discrete(cpt_collection, voxelmodel):
    result = add_nearest_voxelmodel_variable(
        cpt_collection, voxelmodel, ["strat", "lith"]
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (20, 11)
    assert_array_equal(
        result.data[["lith", "strat"]],
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
        ],
    )


@pytest.mark.unittest
def test_add_model_data_voxelmodel_to_layered(borehole_collection, voxelmodel):
    voxelmodel = voxelmodel.rename(
        {"lith": "lithok"}
    )  # Rename variable to avoid name conflict

    # Basic result (`base_result`): add a single DataArray from a model to a Collection
    base_result = add_model_data(borehole_collection, voxelmodel["strat"])
    assert isinstance(base_result, Collection)
    assert base_result.data.shape == (35, 9)
    assert_array_equal(
        base_result.data["strat"],
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
        base_result.data["top"],
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
        base_result.data["bottom"],
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

    # Using a DataFrame should produce the same result as using a Collection
    result_df = add_model_data(borehole_collection.data, voxelmodel["strat"])
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.equals(base_result.data)

    # Specifying "strat" as the data variable should produce the same result for Collection and DataFrame
    result = add_model_data(borehole_collection, voxelmodel, data_vars="strat")
    assert isinstance(result, Collection)
    assert result.data.equals(base_result.data)

    result_df = add_model_data(borehole_collection.data, voxelmodel, data_vars="strat")
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.equals(base_result.data)

    # Using aggregate_vars should also produce the same result for "strat"
    result = add_model_data(
        borehole_collection,
        voxelmodel,
        data_vars="strat",
        aggregate_vars={"lithok": "mean"},
    )
    assert isinstance(result, Collection)
    assert result.data["strat"].equals(base_result.data["strat"])
    assert_array_almost_equal(
        result.data["lithok"],
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
            2.5,
            2.5,
            np.nan,
            np.nan,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            2.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
        ],
    )

    result_df = add_model_data(
        borehole_collection.data,
        voxelmodel,
        data_vars="strat",
        aggregate_vars={"lithok": "mean"},
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df["strat"].equals(base_result.data["strat"])
    assert result_df["lithok"].equals(result.data["lithok"])


@pytest.mark.unittest
def test_add_model_data_multiple_vars(borehole_collection, voxelmodel):
    # Enough to only test with DataFrame here, other behavior is covered in previous tests
    result = add_model_data(
        borehole_collection.data,
        voxelmodel,
        data_vars=["strat", "lith"],
        suffix="_model",
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (37, 10)
    assert result.equals(
        add_model_data(
            borehole_collection.data, voxelmodel[["strat", "lith"]], suffix="_model"
        )
    )
    assert_array_almost_equal(
        result["top"],
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
            2.3,
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
            0.9,
            1.2,
            1.8,
            1.9,
            2.4,
            2.5,
        ],
    )
    assert_array_almost_equal(
        result["bottom"],
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
            2.3,
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
            0.9,
            1.2,
            1.8,
            1.9,
            2.4,
            2.5,
            3.0,
        ],
    )
    assert_array_almost_equal(
        result["strat_model"],
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
            1.0,
            2.0,
            np.nan,
            np.nan,
        ],
    )
    assert_array_almost_equal(
        result["lith_model"],
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
            3.0,
            2.0,
            2.0,
            np.nan,
            np.nan,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            2.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            3.0,
            3.0,
            1.0,
            1.0,
            1.0,
            2.0,
            np.nan,
            np.nan,
        ],
    )


@pytest.mark.unittest
def test_add_model_data_voxelmodel_to_discrete(cpt_collection, voxelmodel):
    result = add_model_data(cpt_collection, voxelmodel["strat"], bottom_="depth")
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
def test_add_model_data_layermodel_to_layered(borehole_collection, layermodel):
    result = add_model_data(
        borehole_collection,
        layermodel,
        data_vars="layer",
        aggregate_vars={"kh": "mean"},
        suffix="_model",
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (38, 10)
    assert_array_equal(
        result.data["layer_model"].fillna("MISSING"),  # Fillna due to ArrowStringArray
        [
            "A",
            "B",
            "B",
            "C",
            "C",
            "C",
            "D",
            "MISSING",
            "MISSING",
            "A",
            "B",
            "B",
            "D",
            "D",
            "D",
            "D",
            "MISSING",
            "A",
            "C",
            "C",
            "C",
            "D",
            "D",
            "MISSING",
            "MISSING",
            "MISSING",
            "MISSING",
            "MISSING",
            "MISSING",
            "MISSING",
            "A",
            "B",
            "B",
            "C",
            "C",
            "C",
            "D",
            "D",
        ],
    )
    assert_array_almost_equal(
        result.data["kh_model"],
        [
            0.04,
            0.2,
            0.2,
            20.1,
            20.1,
            20.1,
            85.0,
            np.nan,
            np.nan,
            0.04,
            0.2,
            0.2,
            85.0,
            85.0,
            85.0,
            85.0,
            np.nan,
            0.04,
            20.1,
            20.1,
            20.1,
            85.0,
            85.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.04,
            0.2,
            0.2,
            20.1,
            20.1,
            20.1,
            85.0,
            85.0,
        ],
    )
    assert_array_almost_equal(
        result.data["top"],
        [
            0.0,
            0.4,
            0.8,
            1.15,
            1.5,
            2.5,
            2.75,
            3.55,
            3.7,
            0.0,
            0.45,
            0.6,
            1.15,
            1.2,
            2.5,
            3.1,
            3.55,
            0.0,
            0.6,
            1.4,
            1.8,
            2.4,
            2.9,
            3.6,
            3.8,
            0.0,
            0.5,
            1.2,
            1.8,
            2.5,
            0.0,
            0.05,
            0.5,
            0.75,
            1.2,
            1.8,
            2.35,
            2.5,
        ],
    )
    assert_array_almost_equal(
        result.data["bottom"],
        [
            0.4,
            0.8,
            1.15,
            1.5,
            2.5,
            2.75,
            3.55,
            3.7,
            4.2,
            0.45,
            0.6,
            1.15,
            1.2,
            2.5,
            3.1,
            3.55,
            3.9,
            0.6,
            1.4,
            1.8,
            2.4,
            2.9,
            3.6,
            3.8,
            5.5,
            0.5,
            1.2,
            1.8,
            2.5,
            3.0,
            0.05,
            0.5,
            0.75,
            1.2,
            1.8,
            2.35,
            2.5,
            3.0,
        ],
    )


@pytest.mark.unittest
def test_add_model_data_layermodel_to_discrete(cpt_collection, layermodel):
    result = add_model_data(
        cpt_collection,
        layermodel,
        data_vars="layer",
        aggregate_vars={"kh": "mean"},
        suffix="_model",
        bottom_="depth",
    )
    assert isinstance(result, Collection)
    assert result.data.shape == (27, 11)
    assert_array_equal(
        result.data["layer_model"].fillna("MISSING"),  # Fillna due to ArrowStringArray
        [
            "A",
            "A",
            "A",
            "B",
            "C",
            "C",
            "C",
            "D",
            "D",
            "MISSING",
            "MISSING",
            "MISSING",
            "MISSING",
            "MISSING",
            "A",
            "A",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "D",
            "D",
            "D",
            "MISSING",
            "MISSING",
        ],
    )
    assert_array_almost_equal(
        result.data["kh_model"],
        [
            0.04,
            0.04,
            0.04,
            0.2,
            20.1,
            20.1,
            20.1,
            85.0,
            85.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.04,
            0.04,
            0.2,
            0.2,
            20.1,
            20.1,
            20.1,
            20.1,
            85.0,
            85.0,
            85.0,
            np.nan,
            np.nan,
        ],
    )
    assert_array_almost_equal(
        result.data["depth"],
        [
            1.0,
            2.0,
            2.25,
            2.95,
            3.0,
            4.0,
            4.55,
            5.0,
            5.35,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            0.5,
            1.0,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.35,
            3.5,
            4.0,
            4.15,
            4.5,
            5.0,
        ],
    )


@pytest.mark.unittest
def test_add_model_data_removes_if_column_is_present(borehole_collection, voxelmodel):
    result_strat = [
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
    ]
    borehole_collection.data["strat"] = 1000  # This column should be replaced

    result = add_model_data(borehole_collection, voxelmodel["strat"])
    assert_array_equal(result.data["strat"], result_strat)

    # When suffix is used it should be kept
    result = add_model_data(borehole_collection, voxelmodel["strat"], suffix="_suffix")
    assert (result.data["strat"] == 1000).all()
    assert_array_equal(result.data["strat_suffix"], result_strat)
