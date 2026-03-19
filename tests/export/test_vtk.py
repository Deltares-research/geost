import numpy as np
import pandas as pd
import pytest
import pyvista as pv
from numpy.testing import assert_array_equal

from geost.export import vtk


@pytest.mark.unittest
def test_prepare_as_continous(cpt_data):
    cpt_prepared = vtk.prepare_as_continuous(
        cpt_data, depth_column="depth", vertical_factor=1
    )
    cpt_prepared_vert_fac = vtk.prepare_as_continuous(
        cpt_data, depth_column="depth", vertical_factor=2
    )

    array_out = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 4.0],
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 6.0],
            [1.0, 1.0, 7.0],
            [1.0, 1.0, 8.0],
            [1.0, 1.0, 9.0],
            [1.0, 1.0, 10.0],
            [2.0, 2.0, 0.5],
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 1.5],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.5],
            [2.0, 2.0, 3.0],
            [2.0, 2.0, 3.5],
            [2.0, 2.0, 4.0],
            [2.0, 2.0, 4.5],
            [2.0, 2.0, 5.0],
        ]
    )

    assert_array_equal(cpt_prepared, array_out)
    assert_array_equal(cpt_prepared_vert_fac, array_out * [1, 1, 2])


@pytest.mark.unittest
def test_prepare_as_layers(borehole_data):
    borehole_prepared = vtk.prepare_as_layers(
        borehole_data, depth_column=["top", "bottom"], vertical_factor=1
    )
    borehole_prepared_vert_fac = vtk.prepare_as_layers(
        borehole_data, depth_column=["top", "bottom"], vertical_factor=2
    )

    array_out = np.array(
        [
            [2.0, 3.0, 0.0],
            [2.0, 3.0, 0.8],
            [2.0, 3.0, 0.8],
            [2.0, 3.0, 1.5],
            [2.0, 3.0, 1.5],
            [2.0, 3.0, 2.5],
            [2.0, 3.0, 2.5],
            [2.0, 3.0, 3.7],
            [2.0, 3.0, 3.7],
            [2.0, 3.0, 4.2],
            [1.0, 4.0, 0.0],
            [1.0, 4.0, 0.6],
            [1.0, 4.0, 0.6],
            [1.0, 4.0, 1.2],
            [1.0, 4.0, 1.2],
            [1.0, 4.0, 2.5],
            [1.0, 4.0, 2.5],
            [1.0, 4.0, 3.1],
            [1.0, 4.0, 3.1],
            [1.0, 4.0, 3.9],
            [4.0, 2.0, 0.0],
            [4.0, 2.0, 1.4],
            [4.0, 2.0, 1.4],
            [4.0, 2.0, 1.8],
            [4.0, 2.0, 1.8],
            [4.0, 2.0, 2.9],
            [4.0, 2.0, 2.9],
            [4.0, 2.0, 3.8],
            [4.0, 2.0, 3.8],
            [4.0, 2.0, 5.5],
            [3.0, 5.0, 0.0],
            [3.0, 5.0, 0.5],
            [3.0, 5.0, 0.5],
            [3.0, 5.0, 1.2],
            [3.0, 5.0, 1.2],
            [3.0, 5.0, 1.8],
            [3.0, 5.0, 1.8],
            [3.0, 5.0, 2.5],
            [3.0, 5.0, 2.5],
            [3.0, 5.0, 3.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.2],
            [1.0, 1.0, 1.2],
            [1.0, 1.0, 1.8],
            [1.0, 1.0, 1.8],
            [1.0, 1.0, 2.5],
            [1.0, 1.0, 2.5],
            [1.0, 1.0, 3.0],
        ]
    )

    assert_array_equal(borehole_prepared, array_out)
    assert_array_equal(borehole_prepared_vert_fac, array_out * [1, 1, 2])
