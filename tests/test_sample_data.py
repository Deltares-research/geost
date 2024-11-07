import pandas as pd
import pytest

import geost


@pytest.mark.unittest
def test_boreholes_usp():
    usp = geost.data.boreholes_usp()
    assert isinstance(usp, geost.base.BoreholeCollection)
    usp = geost.data.boreholes_usp(pandas=True)
    assert isinstance(usp, pd.DataFrame)
