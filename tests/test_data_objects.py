from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pyvista import MultiBlock
from shapely import get_coordinates

from geost.base import BoreholeCollection, PointHeader
from geost.export import geodataclass


class TestLayeredData:
    @pytest.mark.unittest
    def test_to_header(self, borehole_data):
        expected_columns = ["nr", "x", "y", "surface", "end", "geometry"]

        header = borehole_data.to_header()

        assert isinstance(header, PointHeader)
        assert_array_equal(header.gdf.columns, expected_columns)
        assert len(header.gdf) == 5
        assert header["nr"].nunique() == 5

    @pytest.mark.unittest
    def test_to_collection(self, borehole_data):
        collection = borehole_data.to_collection()
        assert isinstance(collection, BoreholeCollection)
        assert isinstance(collection.header, PointHeader)
        assert len(collection.header) == 5

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_data):
        selected = borehole_data.select_by_values("lith", ["V", "K"], how="or")

        expected_nrs = ["A", "B", "C", "D"]
        selected_nrs = selected["nr"].unique()

        assert_array_equal(selected_nrs, expected_nrs)

        selected = borehole_data.select_by_values("lith", ["V", "K"], how="and")

        expected_nrs = ["B", "D"]
        selected_nrs = selected["nr"].unique()

        assert_array_equal(selected_nrs, expected_nrs)

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_data):
        sliced = borehole_data.slice_by_values("lith", "Z")

        expected_boreholes_with_sand = ["A", "C", "D", "E"]
        expected_length = 10

        assert_array_equal(sliced["nr"].unique(), expected_boreholes_with_sand)
        assert np.all(sliced["lith"] == "Z")
        assert len(sliced) == expected_length

        sliced = borehole_data.slice_by_values("lith", "Z", invert=True)

        expected_boreholes_without_sand = ["A", "B", "C", "D"]
        expected_length = 15

        assert_array_equal(sliced["nr"].unique(), expected_boreholes_without_sand)
        assert ~np.any(sliced["lith"] == "Z")
        assert len(sliced) == expected_length

    @pytest.mark.unittest
    def test_get_cumulative_layer_thickness(self, borehole_data):
        result = borehole_data.get_cumulative_layer_thickness("lith", "V")
        expected_boreholes_returned = ["B", "D"]
        expected_thickness = [1.9, 1.4]

        assert len(result) == 2
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["V"], expected_thickness)

        result = borehole_data.get_cumulative_layer_thickness("lith", ["Z", "K"])
        expected_boreholes_returned = ["A", "B", "C", "D", "E"]
        expected_sand_thickness = [2.2, np.nan, 2.6, 0.5, 3.0]
        expected_clay_thickness = [2.0, 2.0, 2.9, 1.1, np.nan]

        assert result.shape == (5, 2)
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["K"], expected_clay_thickness)
        assert_array_almost_equal(result["Z"], expected_sand_thickness)

    @pytest.mark.unittest
    def test_get_layer_top(self, borehole_data):
        result = borehole_data.get_layer_top("lith", "V")
        expected_boreholes_returned = ["B", "D"]
        expected_tops = [1.2, 0.5]

        assert len(result) == 2
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["V"], expected_tops)

        result = borehole_data.get_layer_top("lith", ["Z", "K"])
        expected_boreholes_returned = ["A", "B", "C", "D", "E"]
        expected_sand_top = [1.5, np.nan, 2.9, 2.5, 0.0]
        expected_clay_top = [0.0, 0.0, 0.0, 0.0, np.nan]

        assert result.shape == (5, 2)
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["Z"], expected_sand_top)
        assert_array_almost_equal(result["K"], expected_clay_top)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, borehole_data):
        # Test slicing with respect to depth below the surface.
        upper, lower = 0.6, 2.4
        sliced = borehole_data.slice_depth_interval(upper, lower)

        layers_per_borehole = sliced["nr"].value_counts()
        expected_layer_count = [3, 3, 3, 3, 2]

        assert len(sliced) == 14
        assert sliced["top"].min() == upper
        assert sliced["bottom"].max() == lower
        assert_array_equal(layers_per_borehole, expected_layer_count)

        # Test slicing without updating layer boundaries.
        sliced = borehole_data.slice_depth_interval(
            upper, lower, update_layer_boundaries=False
        )

        expected_tops_of_slice = [0.0, 0.6, 0.0, 0.5, 0.5]
        expected_bottoms_of_slice = [2.5, 2.5, 2.9, 2.5, 2.5]

        tops_of_slice = sliced.df.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.df.groupby("nr")["bottom"].max()

        assert len(sliced) == 14
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        # Test slicing with respect to a vertical reference plane.
        nap_upper, nap_lower = -2, -3
        sliced = borehole_data.slice_depth_interval(
            nap_upper, nap_lower, relative_to_vertical_reference=True
        )

        expected_tops_of_slice = [2.2, 2.3, 2.25, 2.1, 1.9]
        expected_bottoms_of_slice = [3.2, 3.3, 3.25, 3.0, 2.9]

        tops_of_slice = sliced.df.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.df.groupby("nr")["bottom"].max()

        assert len(sliced) == 11
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        # Test slices that return empty objects.
        empty_slice = borehole_data.slice_depth_interval(-2, -1)
        empty_slice_nap = borehole_data.slice_depth_interval(
            3, 2, relative_to_vertical_reference=True
        )

        assert len(empty_slice) == 0
        assert len(empty_slice_nap) == 0

        # Test slicing using only an upper boundary or lower boundary.
        upper = 4
        sliced = borehole_data.slice_depth_interval(upper)

        expected_boreholes = ["A", "C"]

        assert len(sliced) == 2
        assert_array_equal(sliced["nr"], expected_boreholes)

        nap_lower = -0.5
        sliced = borehole_data.slice_depth_interval(
            lower_boundary=nap_lower, relative_to_vertical_reference=True
        )

        bottoms_of_slice = sliced.df.groupby("nr")["bottom"].max()
        expected_bottoms_of_slice = [0.7, 0.8, 0.75, 0.6, 0.4]

        assert len(sliced) == 7
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

    @pytest.mark.unittest
    def test_to_multiblock(self, borehole_data):
        # Test normal to multiblock.
        multiblock = borehole_data.to_multiblock("lith")
        expected_bounds = (0.0, 5.0, 0.0, 6.0, -5.25, 0.3)
        assert isinstance(multiblock, MultiBlock)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds
        assert multiblock[0].n_arrays == 2
        assert multiblock[0].n_cells == 22
        assert multiblock[0].n_points == 160

        # Test with vertical exageration.
        multiblock = borehole_data.to_multiblock("lith", vertical_factor=10)
        expected_bounds = (0.0, 5.0, 0.0, 6.0, -52.5, 3.0)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds

        # Test to multiblock with respect to depth below the surface.
        multiblock = borehole_data.to_multiblock(
            "lith", relative_to_vertical_reference=False
        )
        expected_bounds = (0.0, 5.0, 0.0, 6.0, 0.0, 5.5)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds

        # Test with both options.
        multiblock = borehole_data.to_multiblock(
            "lith", vertical_factor=10, relative_to_vertical_reference=False
        )
        expected_bounds = (0.0, 5.0, 0.0, 6.0, 0.0, 55.0)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds

    @pytest.mark.unittest
    def test_to_datafusiontools(self, borehole_data):
        # Test normal export.
        dft = borehole_data.to_datafusiontools("lith")

        expected_independent_value = [-0.2, -0.95, -1.8, -2.9, -3.75]
        expected_number_of_variables = 1

        assert len(dft) == 5
        assert np.all([isinstance(d, geodataclass.Data) for d in dft])
        assert np.all([len(d.variables) == expected_number_of_variables for d in dft])
        assert_array_almost_equal(
            dft[0].independent_variable.value, expected_independent_value
        )

        # Test with label encoding.
        dft = borehole_data.to_datafusiontools("lith", encode=True)
        expected_number_of_variables = 3
        assert np.all([isinstance(d, geodataclass.Data) for d in dft])
        assert np.all([len(d.variables) == expected_number_of_variables for d in dft])

        # Test without updating layer depths to NAP
        dft = borehole_data.to_datafusiontools(
            "lith", relative_to_vertical_reference=False
        )
        expected_independent_value = [0.4, 1.15, 2.0, 3.1, 3.95]

        assert_array_almost_equal(
            dft[0].independent_variable.value, expected_independent_value
        )

    @pytest.mark.unittest
    def test_to_vtm_with_file(self, borehole_data):
        outfile = Path("temp.vtm")
        outfolder = outfile.parent / r"temp"
        borehole_data.to_vtm(outfile, "lith")
        assert outfile.is_file()
        outfile.unlink()
        for f in outfolder.glob("*.vtp"):
            f.unlink()
        outfolder.rmdir()
        pass

    @pytest.mark.unittest
    def test_to_datafusiontools_with_file(self, borehole_data):
        outfile = Path("dft.pkl")
        borehole_data.to_datafusiontools("lith", outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_qgis3d(self, borehole_data):
        outfile = Path("temp.gpkg")
        borehole_data.to_qgis3d(outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_create_geodataframe_3d(self, borehole_data):
        relative_to_vertical_reference = True
        gdf = borehole_data._create_geodataframe_3d(relative_to_vertical_reference)

        first_line_coords = get_coordinates(gdf['geometry'].iloc[0], include_z=True)
        expected_coords = [[2., 3., 0.21], [2., 3., -0.59]]

        assert all(gdf.geom_type == "LineString")
        assert_array_almost_equal(first_line_coords, expected_coords)

        relative_to_vertical_reference = False
        gdf = borehole_data._create_geodataframe_3d(relative_to_vertical_reference)

        first_line_coords = get_coordinates(gdf['geometry'].iloc[0], include_z=True)
        expected_coords = [[2., 3., 0.01], [2., 3., 0.81]]

        assert all(gdf.geom_type == "LineString")
        assert_array_almost_equal(first_line_coords, expected_coords)

    @pytest.mark.unittest
    def test_change_depth_values(self, borehole_data):
        borehole = borehole_data.select_by_values("nr", "A").df
        borehole = borehole_data._change_depth_values(borehole)

        expected_top = [0.2, -0.6, -1.3, -2.3, -3.5]
        expected_bottom = [-0.6, -1.3, -2.3, -3.5, -4.0]

        assert_array_almost_equal(borehole["top"], expected_top)
        assert_array_almost_equal(borehole["bottom"], expected_bottom)

    @pytest.mark.unittest
    def test_check_correct_instance(self, borehole_data):
        inst = "string"
        inst = borehole_data._check_correct_instance(inst)
        assert isinstance(inst, list)

        inst = ["list of strings"]
        inst = borehole_data._check_correct_instance(inst)
        assert isinstance(inst, list)

    @pytest.mark.unittest
    def test_to_csv_mixin(self, borehole_data):
        outfile = Path("temp.csv")
        borehole_data.to_csv(outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_parquet_mixin(self, borehole_data):
        outfile = Path("temp.parquet")
        borehole_data.to_parquet(outfile)
        assert outfile.is_file()
        outfile.unlink()
