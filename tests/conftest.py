from pathlib import Path

import pytest

from geost import read_sst_cores


@pytest.fixture
def boreholes():
    borehole_file = Path(__file__).parent / "data" / "test_boreholes.parquet"
    borehole_collection = read_sst_cores(borehole_file)
    return borehole_collection
