import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

from pysst.borehole import BoreholeCollection


class TestPointCollection:
    @pytest.fixture
    def borehole_collection(self):
        nr = np.full(10, "B-01")
        x = np.full(10, 139370)
        y = np.full(10, 455540)
        mv = np.full(10, 1.0)
        end = np.full(10, -4.0)
        top = np.array([1, 0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -3.6])
        bottom = np.array([0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -3.6, -4.0])
        data_string = np.array(
            ["K", "Kz", "K", "Ks3", "Ks2", "V", "Zk", "Zs", "Z", "Z"]
        )
        data_int = np.arange(0, 10, dtype=np.int64)
        data_float = np.arange(0, 5, 0.5, dtype=np.float64)

        dataframe = pd.DataFrame(
            {
                "nr": nr,
                "x": x,
                "y": y,
                "mv": mv,
                "end": end,
                "top": top,
                "bottom": bottom,
                "data_string": data_string,
                "data_int": data_int,
                "data_float": data_float,
            }
        )

        return BoreholeCollection(dataframe)

    @pytest.mark.unittest
    def test_change_vertical_reference(self, borehole_collection):
        assert borehole_collection.vertical_reference == "NAP"

        target_top_nap = np.array([1, 0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -3.6])
        target_bottom_nap = np.array(
            [0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -3.6, -4.0]
        )
        target_top_surfacelevel = np.array(
            [0, -0.5, -1, -1.5, -2.5, -3, -3.5, -4, -4.2, -4.6]
        )
        target_bottom_surfacelevel = np.array(
            [-0.5, -1, -1.5, -2.5, -3, -3.5, -4, -4.2, -4.6, -5.0]
        )
        target_top_depth = np.array([0, 0.5, 1, 1.5, 2.5, 3, 3.5, 4, 4.2, 4.6])
        target_bottom_depth = np.array([0.5, 1, 1.5, 2.5, 3, 3.5, 4, 4.2, 4.6, 5.0])

        # From NAP to surfacelevel
        borehole_collection.change_vertical_reference("surfacelevel")
        assert_array_almost_equal(
            borehole_collection.data["top"], target_top_surfacelevel
        )
        assert_array_almost_equal(
            borehole_collection.data["bottom"], target_bottom_surfacelevel
        )

        # From surfacelevel to NAP
        borehole_collection.change_vertical_reference("NAP")
        assert_array_almost_equal(borehole_collection.data["top"], target_top_nap)
        assert_array_almost_equal(borehole_collection.data["bottom"], target_bottom_nap)

        # From NAP to depth
        borehole_collection.change_vertical_reference("depth")
        assert_array_almost_equal(borehole_collection.data["top"], target_top_depth)
        assert_array_almost_equal(
            borehole_collection.data["bottom"], target_bottom_depth
        )

        # From depth to NAP
        borehole_collection.change_vertical_reference("NAP")
        assert_array_almost_equal(borehole_collection.data["top"], target_top_nap)
        assert_array_almost_equal(borehole_collection.data["bottom"], target_bottom_nap)

        # From surfacelevel to depth
        borehole_collection.change_vertical_reference("surfacelevel")
        borehole_collection.change_vertical_reference("depth")
        assert_array_almost_equal(borehole_collection.data["top"], target_top_depth)
        assert_array_almost_equal(
            borehole_collection.data["bottom"], target_bottom_depth
        )

        # From depth to surfacelevel
        borehole_collection.change_vertical_reference("surfacelevel")
        assert_array_almost_equal(
            borehole_collection.data["top"], target_top_surfacelevel
        )
        assert_array_almost_equal(
            borehole_collection.data["bottom"], target_bottom_surfacelevel
        )
