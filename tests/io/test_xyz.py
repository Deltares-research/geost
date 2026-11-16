import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal

from geost.io import xyz


@pytest.mark.unittest
@pytest.mark.parametrize(
    "trim_to_doi, doi_type, end_depth",
    [
        (False, "standard", 487.484),
        (True, "standard", 229.962),
        (True, "conservative", 187.784),
    ],
)
def test_ags_xyz_to_collection(testdatadir, trim_to_doi, doi_type, end_depth):
    file = testdatadir / "test_ags_xyz.xyz"
    collection = xyz.ags_xyz_to_collection(
        file, trim_to_doi=trim_to_doi, doi_type=doi_type
    )
    assert collection is not None
    assert_equal(collection.data["DEP_BOT"].max(), end_depth)
