import pandas as pd
import pytest

import geost
from geost.exceptions import MissingSurveyIDError


@pytest.mark.unittest
def test_pandas_dataframe():
    result = geost.pandas_dataframe(
        {
            "nr": ["A", "A", "B", "B"],
            "top": [0, 0.5, 0, 0.5],
            "bottom": [0.5, 1, 0.5, 1],
            "lith": ["sand", "clay", "sand", "clay"],
        }
    )
    assert isinstance(result, pd.DataFrame)
    assert result.gst._nr == "nr"


@pytest.mark.unittest
def test_pandas_dataframe_error():
    with pytest.raises(MissingSurveyIDError):
        geost.pandas_dataframe({"value": [1, 2]})
