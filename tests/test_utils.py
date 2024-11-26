import pytest

from geost import utils


@pytest.mark.unittest
def test_save_pickle(borehole_collection, tmp_path):
    outfile = tmp_path / r"collection.pkl"
    utils.save_pickle(borehole_collection, outfile)
    assert outfile.is_file()
