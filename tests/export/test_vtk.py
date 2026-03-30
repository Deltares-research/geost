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


@pytest.mark.unittest
def test_generate_cylinders_layereddata(borehole_data):
    cylinders = vtk.generate_cylinders(
        borehole_data,
        depth_column=["top", "bottom"],
        data_columns=["lith"],
        radius=0.5,
        n_sides=8,
        vertical_factor=1,
    )

    # Execute the generator
    cylinders = list(cylinders)

    for cylinder in cylinders:
        assert isinstance(cylinder, pv.PolyData)
        assert cylinder.n_cells == 50
        assert cylinder.n_points == 48
        assert cylinder.cell_data.keys() == ["lith"]


@pytest.mark.unittest
def test_generate_cylinders_continuousdata(cpt_data):
    cylinders = vtk.generate_cylinders(
        cpt_data,
        depth_column="depth",
        data_columns=["qc"],
        radius=0.5,
        n_sides=8,
        vertical_factor=1,
    )

    # Execute the generator
    cylinders = list(cylinders)

    for cylinder in cylinders:
        assert isinstance(cylinder, pv.PolyData)
        assert cylinder.n_cells == 10
        assert cylinder.n_points == 96
        assert list(cylinder.point_data.keys())[0] == "qc"


@pytest.mark.unittest
def test_borehole_to_multiblock(borehole_data, cpt_data):
    multiblock_bh = vtk.borehole_to_multiblock(
        borehole_data,
        depth_column=["top", "bottom"],
        displayed_variables=["lith"],
        radius=0.5,
        n_sides=8,
        vertical_factor=1,
    )

    assert isinstance(multiblock_bh, pv.MultiBlock)
    assert len(multiblock_bh) == 5

    multiblock_cpt = vtk.borehole_to_multiblock(
        cpt_data,
        depth_column="depth",
        displayed_variables=["qc"],
        radius=0.5,
        n_sides=8,
        vertical_factor=1,
    )

    assert isinstance(multiblock_cpt, pv.MultiBlock)
    assert len(multiblock_cpt) == 2


@pytest.mark.unittest
def test_layerdata_to_pyvista_unstructured(borehole_data, cpt_data):
    unstructured_grid_boreholes = vtk.layerdata_to_pyvista_unstructured(
        borehole_data,
        depth_column=["top", "bottom"],
        displayed_variables=["lith"],
    )

    assert isinstance(unstructured_grid_boreholes, pv.UnstructuredGrid)
    assert unstructured_grid_boreholes.n_cells == len(borehole_data)
    assert unstructured_grid_boreholes.n_points == 200
    assert unstructured_grid_boreholes.cell_data.keys() == ["lith"]

    unstructured_grid_cpt = vtk.layerdata_to_pyvista_unstructured(
        cpt_data, depth_column="depth", displayed_variables=["qc"]
    )

    assert isinstance(unstructured_grid_cpt, pv.UnstructuredGrid)
    assert unstructured_grid_cpt.n_cells == len(cpt_data)
    assert unstructured_grid_cpt.n_points == 160
    assert unstructured_grid_cpt.cell_data.keys() == ["qc"]
