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

@pytest.mark.unittest
def test_config_validation_flag_invalid():
    # Modify settings to non-default values
    config.validation.DROP_INVALID = True
    config.validation.FLAG_INVALID = False

    # Set FLAG_INVALID to True, which should automatically set DROP_INVALID to False
    config.validation.FLAG_INVALID = True
    assert config.validation.FLAG_INVALID is True
    assert config.validation.DROP_INVALID is False

    # Set FLAG_INVALID to False, which should restore the user's preference for DROP_INVALID
    config.validation.FLAG_INVALID = False
    assert config.validation.FLAG_INVALID is False
    assert config.validation.DROP_INVALID is True
