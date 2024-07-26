from pathlib import Path

import pandas as pd
import pytest

from geost import read_sst_cores
from geost.new_base import LayeredData


@pytest.fixture
def borehole_file():
    return Path(__file__).parent / "data" / "test_boreholes.parquet"


@pytest.fixture
def borehole_collection(borehole_file):
    borehole_collection = read_sst_cores(borehole_file)
    return borehole_collection


@pytest.fixture
def borehole_data(borehole_file):
    df = pd.read_parquet(borehole_file)
    return LayeredData(df)
