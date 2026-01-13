import pytest

from geost import config


@pytest.mark.unittest
def test_config_validation_reset():
    # Modify settings to non-default values
    config.validation.VERBOSE = False
    config.validation.DROP_INVALID = False
    config.validation.FLAG_INVALID = True
    config.validation.AUTO_ALIGN = False

    config.validation.reset_settings()

    assert config.validation.VERBOSE is True
    assert config.validation.DROP_INVALID is True
    assert config.validation.FLAG_INVALID is False
    assert config.validation.AUTO_ALIGN is True
