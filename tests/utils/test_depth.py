import pandas as pd
import pytest

from geost.utils import depth


@pytest.fixture
def inconsistent_layered():
    return pd.DataFrame(
        {
            "nr": ["BH1", "BH1", "BH1", "BH1"],
            "top": [0, 10, 10, 20],
            "bottom": [10, 15, 20, 30],
        }
    )


@pytest.mark.unittest
def test_reset_tops(inconsistent_layered):
    result = depth.reset_tops(inconsistent_layered, nr="nr", top="top", bottom="bottom")

    expected = pd.DataFrame(
        {
            "nr": ["BH1", "BH1", "BH1", "BH1"],
            "top": [0, 10, 15, 20],
            "bottom": [10, 15, 20, 30],
        }
    )

    pd.testing.assert_frame_equal(result, expected)
