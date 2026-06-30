import pytest
from numpy.testing import assert_almost_equal

from geost.utils import unit_conversion


@pytest.mark.unittest
def test_parse_unit_expression():
    assert unit_conversion.parse_unit_expression("mm") == {
        "left": "mm",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("cm") == {
        "left": "cm",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("m") == {
        "left": "m",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("km") == {
        "left": "km",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("s") == {
        "left": "s",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("min") == {
        "left": "min",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("h") == {
        "left": "h",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("d") == {
        "left": "d",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("w") == {
        "left": "w",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("yr") == {
        "left": "yr",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("API") == {
        "left": "API",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("gAPI") == {
        "left": "gAPI",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("mAPI") == {
        "left": "mAPI",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("uAPI") == {
        "left": "uAPI",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("ohm") == {
        "left": "ohm",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("kohm") == {
        "left": "kohm",
        "right": None,
        "operator": None,
    }
    assert unit_conversion.parse_unit_expression("m/s") == {
        "left": "m",
        "right": "s",
        "operator": "/",
    }
    assert unit_conversion.parse_unit_expression("m per s") == {
        "left": "m",
        "right": "s",
        "operator": "per",
    }
    assert unit_conversion.parse_unit_expression("ohm m") == {
        "left": "ohm",
        "right": "m",
        "operator": " ",
    }


def test_calculate_factor():
    assert_almost_equal(unit_conversion.calculate_factor("mm", "cm"), 0.1)
    assert_almost_equal(unit_conversion.calculate_factor("cm", "mm"), 10)
    assert_almost_equal(unit_conversion.calculate_factor("m", "km"), 0.001)
    assert_almost_equal(unit_conversion.calculate_factor("km", "m"), 1000)
    assert_almost_equal(unit_conversion.calculate_factor("s", "min"), 1 / 60)
    assert_almost_equal(unit_conversion.calculate_factor("min", "s"), 60)
    assert_almost_equal(unit_conversion.calculate_factor("h", "d"), 1 / 24)
    assert_almost_equal(unit_conversion.calculate_factor("d", "h"), 24)
    assert_almost_equal(
        unit_conversion.calculate_factor("w", "yr"), 1 / 52.1429, decimal=4
    )
    assert_almost_equal(unit_conversion.calculate_factor("yr", "w"), 52.1429, decimal=4)
    assert_almost_equal(unit_conversion.calculate_factor("API", "gAPI"), 1)
    assert_almost_equal(unit_conversion.calculate_factor("gAPI", "API"), 1)
    assert_almost_equal(unit_conversion.calculate_factor("mAPI", "uAPI"), 1000000)
    assert_almost_equal(unit_conversion.calculate_factor("uAPI", "mAPI"), 0.000001)
    assert_almost_equal(unit_conversion.calculate_factor("ohm", "kohm"), 0.001)
    assert_almost_equal(unit_conversion.calculate_factor("kohm", "ohm"), 1000)
    assert_almost_equal(unit_conversion.calculate_factor("m/s", "km/h"), 3.6)
    assert_almost_equal(
        unit_conversion.calculate_factor("km/h", "m/s"), 1 / 3.6, decimal=4
    )
    assert_almost_equal(unit_conversion.calculate_factor("m/s", "cm/s"), 100)
    assert_almost_equal(unit_conversion.calculate_factor("cm/s", "m/s"), 0.01)
    assert_almost_equal(unit_conversion.calculate_factor("m/min", "m/s"), 1 / 60)
    assert_almost_equal(unit_conversion.calculate_factor("m/s", "m/min"), 60)
    assert_almost_equal(unit_conversion.calculate_factor("km/s", "m/s"), 1000)
    assert_almost_equal(unit_conversion.calculate_factor("m/s", "km/s"), 0.001)
    assert_almost_equal(unit_conversion.calculate_factor("ohm.m", "ohm.cm"), 100)
    assert_almost_equal(unit_conversion.calculate_factor("ohm cm", "ohm m"), 0.01)
    assert_almost_equal(unit_conversion.calculate_factor("kohmxm", "ohmxm"), 1000)
    assert_almost_equal(unit_conversion.calculate_factor("ohm m", "kohm m"), 0.001)
    assert_almost_equal(unit_conversion.calculate_factor("m per s", "km per h"), 3.6)
    assert_almost_equal(
        unit_conversion.calculate_factor("km per h", "m per s"), 1 / 3.6, decimal=4
    )
    assert_almost_equal(unit_conversion.calculate_factor("mm/s", "um/s"), 1000)
    assert_almost_equal(unit_conversion.calculate_factor("um/s", "mm/s"), 0.001)
    assert_almost_equal(unit_conversion.calculate_factor("m over h", "m over d"), 24)
    assert_almost_equal(unit_conversion.calculate_factor("m/d", "m/h"), 1 / 24)
    assert_almost_equal(
        unit_conversion.calculate_factor("API per m", "API per cm"), 0.01
    )
    assert_almost_equal(
        unit_conversion.calculate_factor("API per cm", "API per m"), 100
    )
