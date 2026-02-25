import pytest

from geost.analysis import layers


@pytest.mark.unittest
def test_get_layer_top(borehole_data):
    layers.get_layer_top(borehole_data, "lith", "Z")
