import pytest

from geost.download.pdok_geopackage import get_geopackage_url


@pytest.mark.unittest
def test_get_geopackage_url_bhrgt():
    result = get_geopackage_url("BHR-GT")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/geotechnischbooronderzoek.zip")


@pytest.mark.unittest
def test_get_geopackage_url_bhrg():
    result = get_geopackage_url("BHR-G")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brobhrgvolledigeset.gpkg")


@pytest.mark.unittest
def test_get_geopackage_url_bhrp():
    result = get_geopackage_url("BHR-P")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brobhrpvolledigeset.zip")


@pytest.mark.unittest
def test_get_geopackage_url_cpt():
    result = get_geopackage_url("CPT")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brocptvolledigeset_v2_0.zip")


@pytest.mark.unittest
def test_get_geopackage_url_sfr():
    result = get_geopackage_url("SFR")
    assert isinstance(result, str)
    assert result.endswith("atom/downloads/brosfrvolledigeset.zip")


@pytest.mark.unittest
def test_get_geopackage_url_invalid_type():
    with pytest.raises(KeyError, match="Unknown dataset type: invalid_type"):
        get_geopackage_url("invalid_type")
