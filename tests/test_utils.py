import pytest
from pyogrio.errors import FieldError

from geost import utils


@pytest.mark.unittest
def test_save_pickle(borehole_collection, tmp_path):
    outfile = tmp_path / r"collection.pkl"
    utils.save_pickle(borehole_collection, outfile)
    assert outfile.is_file()


@pytest.mark.unittest
def test_to_geopackage(borehole_collection, tmp_path):
    outfile = tmp_path / "gpkg_with_error.gpkg"
    invalid_column = "GEOM"
    borehole_collection.header[invalid_column] = "test"
    with pytest.raises(FieldError):
        borehole_collection.to_geopackage(outfile)
