import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import Point

from geost import Collection, config
from geost._warnings import AlignmentWarning, ValidationWarning
from geost.validation import ValidationResult


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
                        "nr": ["a", "b", "b"],
                        "column": [1, 4, 5],
                    },
                    index=pd.Index([0, 3, 4]),
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
    )
    def test_handle_errors(self, drop_invalid, flag_invalid, result_df, test_df):
        validation = ValidationResult()
        validation.add("column", "message", ["a"], [1, 2])

        config.validation.DROP_INVALID = drop_invalid
        config.validation.FLAG_INVALID = flag_invalid

        validation.handle_errors(test_df)
        assert test_df.equals(result_df)
