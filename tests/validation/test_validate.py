import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost import config
from geost._warnings import ValidationWarning
from geost.validation import ValidationResult, validate


class TestValidationResult:
    @pytest.fixture
    def test_df(self):
        data = pd.DataFrame(
            {
                "nr": ["a", "a", "a", "b", "b"],
                "column": [1, 2, 3, 4, 5],
            }
        )
        return data

    @pytest.mark.unittest
    def test_add(self):
        validation = ValidationResult()
        validation.add("column", "message", ["a"], [1, 2])
        assert validation.has_errors
        assert validation.error_nrs == ["a"]
        assert_array_equal(validation.error_indices, pd.Index([1, 2]))

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "drop_invalid, flag_invalid",
        [
            (True, False),  # CASE I: drop invalid, do not flag
            (False, True),  # CASE II: do not drop invalid, flag them
            (False, False),  # CASE III: do not drop invalid, do not flag them
        ],
        ids=["drop", "flag", "skip"],
    )
    def test_display_warnings(self, drop_invalid, flag_invalid):
        validation = ValidationResult()
        validation.add("column", "message", ["a"], [1, 2])

        config.validation.DROP_INVALID = drop_invalid
        config.validation.FLAG_INVALID = flag_invalid

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validation.display_warnings()
            assert len(w) == 1
            assert issubclass(w[-1].category, ValidationWarning)
            assert "VALIDATION ISSUE (1/1)" in str(w[-1].message)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "drop_invalid, flag_invalid, result_df",
        [
            (
                True,
                False,
                pd.DataFrame(
                    {
                        "nr": ["b", "b"],
                        "column": [4, 5],
                    },
                    index=pd.Index([3, 4]),
                ),
            ),  # CASE I: drop invalid, do not flag
            (
                False,
                True,
                pd.DataFrame(
                    {
                        "nr": ["a", "a", "a", "b", "b"],
                        "column": [1, 2, 3, 4, 5],
                        "is_valid": [True, False, False, True, True],
                    }
                ),
            ),  # CASE II: do not drop invalid, flag them
            (
                False,
                False,
                pd.DataFrame(
                    {
                        "nr": ["a", "a", "a", "b", "b"],
                        "column": [1, 2, 3, 4, 5],
                    }
                ),
            ),  # CASE III: do not drop invalid, do not flag them
        ],
        ids=["drop", "flag", "skip"],
    )
    def test_handle_errors(self, drop_invalid, flag_invalid, result_df, test_df):
        validation = ValidationResult()
        validation.add("column", "message", ["a"], [1, 2])

        config.validation.DROP_INVALID = drop_invalid
        config.validation.FLAG_INVALID = flag_invalid

        test_df = validation.handle_errors(test_df, nr_col="nr")
        assert test_df.equals(result_df)


class TestValidationFunctions:
    @pytest.fixture
    def column_names_top_bot(self):
        return {
            "nr": "nr",
            "surface": "surface",
            "x": "x",
            "y": "y",
            "top": "top",
            "depth": "bottom",
        }

    @pytest.fixture
    def column_names_depth(self):
        return {
            "nr": "nr",
            "surface": "surface",
            "x": "x",
            "y": "y",
            "top": None,
            "depth": "depth",
        }

    @pytest.fixture
    def base_df(self):
        df = pd.DataFrame(
            {
                "nr": ["a", "a", "a", "b", "b"],
                "surface": [1, 1, 1, 2, 2],
            }
        )
        return df

    @pytest.fixture
    def invalid_base_df(self):
        df = pd.DataFrame(
            {
                "nr": ["a", "a", "a", "b", "b"],
                "surface": ["a", "a", "a", 2, 2],
            }
        )
        return df

    @pytest.fixture
    def valid_dtypes(self, base_df):
        df = base_df.copy()
        df["x"] = [1, 2, 3, 4, 5]
        df["y"] = [1, 2, 3, 4, 5]
        return df

    @pytest.fixture
    def invalid_dtypes_df(self, invalid_base_df):
        df = invalid_base_df.copy()
        df["x"] = [1, "a", 3, 4, 5]
        df["y"] = [1, 2, np.nan, 4, 5]
        return df

    @pytest.fixture
    def valid_top_bottom_df(self, base_df):
        df = base_df.copy()
        df["top"] = [0, 1, 2, 0, 2]
        df["bottom"] = [1, 2, 3, 2, 4]
        return df

    @pytest.fixture
    def invalid_top_bottom_df(self, base_df):
        df = base_df.copy()
        df["top"] = [0, 1, 2, 1, 2]
        df["bottom"] = [1, 0.5, 3, 2, 4]
        return df

    @pytest.fixture
    def valid_depth_df(self, base_df):
        df = base_df.copy()
        df["depth"] = [0, 1, 2, 0, 2]
        return df

    @pytest.fixture
    def invalid_depth_df(self, base_df):
        df = base_df.copy()
        df["depth"] = [0, 1, -2, 0, 2]
        return df

    @pytest.fixture
    def full_top_bottom_df(self, base_df):
        df = base_df.copy()
        df["x"] = [1, 2, 3, 4, 5]
        df["y"] = [1, 2, 3, 4, 5]
        df["top"] = [0, 1, 2, 0, 2]
        df["bottom"] = [1, 2, 3, 2, 4]
        return df

    @pytest.fixture
    def full_depth_df(self, base_df):
        df = base_df.copy()
        df["x"] = [1, 2, 3, 4, 5]
        df["y"] = [1, 2, 3, 4, 5]
        df["depth"] = [0, 1, 2, 0, 2]
        return df

    @pytest.mark.unittest
    def test_coerce_numeric(self, valid_dtypes, invalid_dtypes_df, column_names_depth):
        validation_result = ValidationResult()

        columns_to_check = [
            column_names_depth["surface"],
            column_names_depth["x"],
            column_names_depth["y"],
        ]

        # Valid
        validate.coerce_numeric(valid_dtypes, columns_to_check, validation_result)
        assert not validation_result.has_errors

        # Invalid - non-numeric x, surface levels and NaN y
        validate.coerce_numeric(invalid_dtypes_df, columns_to_check, validation_result)
        validation_result.handle_errors(invalid_dtypes_df, nr_col="nr")
        assert validation_result.has_errors
        assert all(validation_result.errors["surface"]["indices"] == [0, 1, 2])
        assert all(validation_result.errors["x"]["indices"] == [1])
        assert all(validation_result.errors["y"]["indices"] == [2])

    @pytest.mark.unittest
    def test_validate_base(self, base_df, invalid_base_df, column_names_depth):
        validate.validate_base(base_df, column_names_depth)
        validate.validate_base(invalid_base_df, column_names_depth)

    @pytest.mark.unittest
    def test_validate_top_bottom(
        self, valid_top_bottom_df, invalid_top_bottom_df, column_names_top_bot
    ):
        validation_result = ValidationResult()

        # Valid
        validate.validate_top_bottom(
            valid_top_bottom_df,
            column_names_top_bot,
            valid_top_bottom_df["nr"] != valid_top_bottom_df["nr"].shift(),
            validation_result,
        )
        assert not validation_result.has_errors

        # Invalid - non-increasing bottom value and one top that doesn't start at 0
        validate.validate_top_bottom(
            invalid_top_bottom_df,
            column_names_top_bot,
            invalid_top_bottom_df["nr"] != invalid_top_bottom_df["nr"].shift(),
            validation_result,
        )
        validation_result.handle_errors(invalid_top_bottom_df, nr_col="nr")
        assert validation_result.has_errors
        assert all(validation_result.errors["top, bottom"]["indices"] == [1])

    @pytest.mark.unittest
    def test_validate_depth(self, valid_depth_df, invalid_depth_df, column_names_depth):
        validation_result = ValidationResult()

        # Valid
        validate.validate_depths(
            valid_depth_df,
            column_names_depth,
            valid_depth_df["nr"] != valid_depth_df["nr"].shift(),
            validation_result,
        )
        assert not validation_result.has_errors

        # Invalid - 1 x non-increasing depth value
        validate.validate_depths(
            invalid_depth_df,
            column_names_depth,
            invalid_depth_df["nr"] != invalid_depth_df["nr"].shift(),
            validation_result,
        )
        validation_result.handle_errors(invalid_depth_df, nr_col="nr")
        assert validation_result.has_errors
        assert all(validation_result.errors["depth"]["indices"] == [2])

    @pytest.mark.unittest
    def test_full_validation(self, full_top_bottom_df, full_depth_df):
        # Top / bottom
        df_tb, result_tb = validate.validate_geostframe(
            full_top_bottom_df,
            has_depth_columns=False,
            is_layered=True,
            first_row_in_survey=full_top_bottom_df["nr"]
            != full_top_bottom_df["nr"].shift(),
            positional_columns=full_top_bottom_df.gst.positional_columns,
        )

        # Depth
        df_d, result_d = validate.validate_geostframe(
            full_depth_df,
            has_depth_columns=True,
            is_layered=False,
            first_row_in_survey=full_depth_df["nr"] != full_depth_df["nr"].shift(),
            positional_columns=full_depth_df.gst.positional_columns,
        )

        assert len(result_tb) == 0
        assert len(result_d) == 0
