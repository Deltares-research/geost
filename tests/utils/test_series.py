import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost.utils import series


@pytest.mark.parametrize(
    "values, expected",
    [
        (1, [True, False, False, False, False]),
        ([2, 4], [False, True, False, True, False]),
        (slice(2, 4), [False, True, True, True, False]),
    ],
)
def test_mask(values, expected):
    s = pd.Series([1, 2, 3, 4, 5])
    assert_array_equal(series.mask(values, s), expected)


@pytest.mark.parametrize(
    "values",
    [
        [1, 1, 2, 2, 2, 1, 1, 3],
        ["Z", "Z", "L", "L", "L", "Z", "Z", "K"],
        [False, False, True, True, True, False, False, True],
    ],
)
def test_label_consecutive_elements(values):
    s = pd.Series(values)
    assert_array_equal(series.label_consecutive_elements(s), [1, 1, 2, 2, 2, 3, 3, 4])
