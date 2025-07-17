from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pyvista import MultiBlock, UnstructuredGrid
from shapely import get_coordinates

from geost.base import (
    BoreholeCollection,
    CptCollection,
    DiscreteData,
    LayeredData,
    PointHeader,
)
from geost.data_objects import Cpt
from geost.export import geodataclass


class TestLayeredData:
    @pytest.mark.unittest
    def test_datatype(self, borehole_data):
        assert borehole_data.datatype == "layered"

    @pytest.mark.unittest
    def test_to_header(self, borehole_data):
        expected_columns = ["nr", "x", "y", "surface", "end", "geometry"]

        header = borehole_data.to_header()

        assert isinstance(header, PointHeader)
        assert_array_equal(header.gdf.columns, expected_columns)
        assert len(header.gdf) == 5
        assert header["nr"].nunique() == 5
        assert header.horizontal_reference == 28992
        assert header.vertical_reference == 5709

    @pytest.mark.unittest
    def test_to_collection(self, borehole_data):
        collection = borehole_data.to_collection()
        assert isinstance(collection, BoreholeCollection)
        assert isinstance(collection.header, PointHeader)
        assert len(collection.header) == 5

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_data):
        selected = borehole_data.select_by_values("lith", ["V", "K"], how="or")
        assert isinstance(selected, LayeredData)

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
        assert isinstance(sliced, LayeredData)

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
    def test_get_cumulative_thickness(self, borehole_data):
        result = borehole_data.get_cumulative_thickness("lith", "V")
        expected_boreholes_returned = ["B", "D"]
        expected_thickness = [1.9, 1.4]

        assert len(result) == 2
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["V"], expected_thickness)

        result = borehole_data.get_cumulative_thickness("lith", ["Z", "K"])
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

        # Test with a minimum depth to start the search for the top of the layer.
        result = borehole_data.get_layer_top("lith", "V", min_depth=2)
        expected_boreholes_returned = ["B", "D"]
        expected_tops = [2.0, 2.0]

        assert len(result) == 2
        assert_array_equal(result.index, expected_boreholes_returned)
        assert_array_almost_equal(result["V"], expected_tops)

        # Test with a minimum depth and minimum thickness
        result = borehole_data.get_layer_top(
            "lith", "V", min_depth=2, min_thickness=0.6
        )
        expected_boreholes_returned = ["B"]
        expected_tops = [2.5]

        assert len(result) == 1
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
        assert isinstance(sliced, LayeredData)

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
    def test_select_by_condition(self, borehole_data):
        selected = borehole_data.select_by_condition(borehole_data["lith"] == "V")
        assert isinstance(selected, LayeredData)

        expected_nrs = ["B", "D"]
        assert_array_equal(selected["nr"].unique(), expected_nrs)
        assert np.all(selected["lith"] == "V")
        assert len(selected) == 4

        selected = borehole_data.select_by_condition(
            borehole_data["lith"] == "V", invert=True
        )
        assert len(selected) == 21
        assert ~np.all(selected["lith"] == "V")

    @pytest.mark.unittest
    def test_to_pyvista_cylinders(self, borehole_data):
        # Test normal to multiblock.
        multiblock = borehole_data.to_pyvista_cylinders("lith")
        expected_bounds = (0.0, 5.0, 0.0, 6.0, -5.25, 0.3)
        assert isinstance(multiblock, MultiBlock)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds
        assert multiblock[0].n_arrays == 2
        assert multiblock[0].n_cells == 22
        assert multiblock[0].n_points == 160

        # Test with vertical exageration.
        multiblock = borehole_data.to_pyvista_cylinders("lith", vertical_factor=10)
        expected_bounds = (0.0, 5.0, 0.0, 6.0, -52.5, 3.0)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds

        # Test to multiblock with respect to depth below the surface.
        multiblock = borehole_data.to_pyvista_cylinders(
            "lith", relative_to_vertical_reference=False
        )
        expected_bounds = (0.0, 5.0, 0.0, 6.0, 0.0, 5.5)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds

        # Test with both options.
        multiblock = borehole_data.to_pyvista_cylinders(
            "lith", vertical_factor=10, relative_to_vertical_reference=False
        )
        expected_bounds = (0.0, 5.0, 0.0, 6.0, 0.0, 55.0)
        assert multiblock.n_blocks == 5
        assert multiblock.bounds == expected_bounds

    @pytest.mark.unittest
    def test_to_pyvista_grid(self, borehole_data):
        grid = borehole_data.to_pyvista_grid(["lith"], radius=0.1)
        assert isinstance(grid, UnstructuredGrid)
        assert grid.n_cells > 0
        assert "lith" in grid.array_names

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
    def test_to_datafusiontools_with_file(self, borehole_data):
        outfile = Path("dft.pkl")
        borehole_data.to_datafusiontools("lith", outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_qgis3d(self, borehole_data):
        outfile = Path("temp.gpkg")
        borehole_data.to_qgis3d(outfile, crs=28992)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_kingdom(self, borehole_data):
        outfile = Path("temp_kingdom.csv")
        tdfile = Path(outfile.parent, f"{outfile.stem}_TDCHART{outfile.suffix}")
        borehole_data.to_kingdom(outfile)
        assert outfile.is_file()
        assert tdfile.is_file()
        out_layerdata = pd.read_csv(outfile)
        out_tddata = pd.read_csv(tdfile)
        assert_array_almost_equal(
            out_layerdata["Start depth"][:6],
            np.array([0.0, 0.8, 1.5, 2.5, 3.7, 0.0]),
        )
        assert_array_almost_equal(
            out_layerdata["End depth"][:6],
            np.array([0.8, 1.5, 2.5, 3.7, 4.2, 0.6]),
        )
        assert_array_almost_equal(
            out_layerdata["Total depth"][:6],
            np.array([4.2, 4.2, 4.2, 4.2, 4.2, 3.9]),
        )
        assert_array_almost_equal(
            out_tddata["MD"],
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        )
        assert_array_almost_equal(
            out_tddata["TWT"],
            np.array(
                [
                    -0.26666667,
                    0.98333333,
                    -0.4,
                    0.85,
                    -0.33333333,
                    0.91666667,
                    -0.13333333,
                    1.11666667,
                    0.13333333,
                    1.38333333,
                ]
            ),
        )
        outfile.unlink()
        tdfile.unlink()

    @pytest.mark.unittest
    def test_create_geodataframe_3d(self, borehole_data):
        relative_to_vertical_reference = True
        gdf = borehole_data._create_geodataframe_3d(
            relative_to_vertical_reference, crs=28992
        )

        first_line_coords = get_coordinates(gdf["geometry"].iloc[0], include_z=True)
        expected_coords = [[2.0, 3.0, 0.21], [2.0, 3.0, -0.59]]

        assert all(gdf.geom_type == "LineString")
        assert_array_almost_equal(first_line_coords, expected_coords)

        relative_to_vertical_reference = False
        gdf = borehole_data._create_geodataframe_3d(
            relative_to_vertical_reference, crs=28992
        )

        first_line_coords = get_coordinates(gdf["geometry"].iloc[0], include_z=True)
        expected_coords = [[2.0, 3.0, 0.01], [2.0, 3.0, 0.81]]

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


class TestDiscreteData:
    @pytest.mark.unittest
    def test_datatype(self, cpt_data):
        assert cpt_data.datatype == "discrete"

    @pytest.mark.unittest
    def test_to_header(self, cpt_data):
        header = cpt_data.to_header()
        expected_columns = ["nr", "x", "y", "surface", "end", "geometry"]

        header = cpt_data.to_header()

        assert isinstance(header, PointHeader)
        assert_array_equal(header.gdf.columns, expected_columns)
        assert len(header.gdf) == 2
        assert header["nr"].nunique() == 2
        assert header.horizontal_reference == 28992
        assert header.vertical_reference == 5709

    @pytest.mark.unittest
    def test_to_collection(self, cpt_data):
        collection = cpt_data.to_collection()
        assert isinstance(collection, CptCollection)
        assert isinstance(collection.header, PointHeader)
        assert len(collection.header) == 2

    @pytest.mark.unittest
    def test_select_by_values(self, cpt_data):
        selected = cpt_data.select_by_values("nr", "a")
        assert isinstance(selected, DiscreteData)
        assert len(selected) == 10
        assert_array_equal(selected["nr"].unique(), "a")

    @pytest.mark.unittest
    def test_select_by_condition(self, cpt_data):
        selected = cpt_data.select_by_condition(
            (cpt_data["qc"] > 0.3) & (cpt_data["qc"] < 0.4)
        )
        assert isinstance(selected, DiscreteData)
        assert len(selected) == 5
        assert np.all((selected["qc"] > 0.3) & (selected["qc"] < 0.4))
        assert_array_equal(selected.df.index, [2, 3, 4, 5, 6])

        selected = cpt_data.select_by_condition(
            (cpt_data["qc"] < 0.35) & (cpt_data["fs"] < 0.2)
        )
        assert len(selected) == 3
        assert np.all(selected["qc"] < 0.35)
        assert np.all(selected["fs"] < 0.2)
        assert_array_equal(selected.df.index, [0, 1, 2])

    @pytest.mark.unittest
    def test_slice_depth_interval(self, cpt_data):
        # Selection with respect to surface level
        selected = cpt_data.slice_depth_interval(2, 3)
        assert isinstance(selected, DiscreteData)
        assert len(selected) == 4
        assert_array_equal(selected["depth"], [2, 3, 2, 3])

        # Selection with respect to vertical reference plane
        selected = cpt_data.slice_depth_interval(
            1.9, 0.9, relative_to_vertical_reference=True
        )
        assert len(selected) == 1
        assert_array_equal(selected["depth"], [1])
        assert_array_equal(selected["nr"], ["a"])

        # Selection with respect to surface level using one limit
        selected = cpt_data.slice_depth_interval(lower_boundary=0.9)
        assert len(selected) == 2
        assert_array_equal(selected["depth"], [0, 0])
        assert_array_equal(selected["nr"], ["a", "b"])

        # Selection with respect to vertical reference plane using one limit
        selected = cpt_data.slice_depth_interval(
            lower_boundary=0.9, relative_to_vertical_reference=True
        )
        assert len(selected) == 2
        assert_array_equal(selected["depth"], [0, 1])
        assert np.all(selected["nr"] == "a")


class TestDataclasses:
    @pytest.mark.unittest
    def test_cpt_dataclass(self):
        cpt = Cpt()
        assert isinstance(cpt, Cpt)
        assert cpt.nr is None
        assert cpt.x is None
        assert cpt.y is None
        assert cpt.z is None
        assert cpt.enddepth is None
        assert cpt.gefid is None
        assert cpt.ncolumns is None
        assert cpt.columninfo is None
        assert cpt.companyid is None
        assert cpt.filedate is None
        assert cpt.fileowner is None
        assert cpt.lastscan is None
        assert cpt.procedurecode is None
        assert cpt.reportcode is None
        assert cpt.projectid is None
