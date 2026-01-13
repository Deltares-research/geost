from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import pyvista as pv
import xarray as xr
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
from shapely.geometry import LineString, Point, Polygon

from geost.base import BoreholeCollection
from geost.export import geodataclass


class TestCollection:
    @pytest.fixture
    def update_raster(self):
        x_coors = np.arange(127000, 128500, 500)
        y_coors = np.arange(503000, 501500, -500)
        data = np.ones((3, 3))
        array = xr.DataArray(data, {"x": x_coors, "y": y_coors})
        return array

    @pytest.mark.unittest
    def test_get(self, borehole_collection):
        selection = borehole_collection.get("A")
        assert selection.header.iloc[0, 0] == "A"
        assert_array_equal(selection.data["nr"].unique(), ["A"])

        selection = borehole_collection.get(["A", "B"])
        assert list(selection.header["nr"].unique()) == ["A", "B"]
        assert_array_equal(selection.data["nr"].unique(), ["A", "B"])

    @pytest.mark.unittest
    def test_add_header_column_to_data(self, borehole_collection):
        borehole_collection.header["test_data"] = [
            i for i in range(len(borehole_collection.header))
        ]
        borehole_collection.add_header_column_to_data("test_data")

        assert_allclose(borehole_collection.get("A").data["test_data"], 0)
        assert_allclose(borehole_collection.get("B").data["test_data"], 1)

    @pytest.mark.unittest
    def test_change_vertical_reference(self, borehole_collection):
        assert borehole_collection.vertical_reference == 5709
        borehole_collection.change_vertical_reference("Ostend height")
        assert borehole_collection.vertical_reference == 5710

    @pytest.mark.unittest
    def test_change_horizontal_reference(self, borehole_collection):
        assert borehole_collection.horizontal_reference == 28992
        borehole_collection.change_horizontal_reference(32631)
        assert borehole_collection.horizontal_reference == 32631
        assert_almost_equal(
            borehole_collection.data["x"][0], borehole_collection.header["x"][0]
        )
        assert_almost_equal(borehole_collection.data["x"][0], 523403.281803, decimal=5)
        assert_almost_equal(
            borehole_collection.data["y"][0], borehole_collection.header["y"][0]
        )
        assert_almost_equal(borehole_collection.data["y"][0], 5313546.187669, decimal=5)

    @pytest.mark.unittest
    def test_change_horizontal_reference_with_inclined(self, nlog_borehole_collection):
        assert nlog_borehole_collection.horizontal_reference == 28992
        nlog_borehole_collection.change_horizontal_reference(32631)
        assert nlog_borehole_collection.horizontal_reference == 32631
        assert_almost_equal(
            nlog_borehole_collection.header["x"][0], 532641.76, decimal=2
        )
        assert_almost_equal(
            nlog_borehole_collection.header["y"][0], 6.170701e6, decimal=0
        )
        assert_almost_equal(
            nlog_borehole_collection.data["x_bot"][0], 532650.77, decimal=2
        )
        assert_almost_equal(
            nlog_borehole_collection.data["y_bot"][0], 6.170700e6, decimal=0
        )

    @pytest.mark.unittest
    def test_reset_header(self, borehole_collection):
        borehole_collection.reset_header()
        assert borehole_collection.horizontal_reference == 28992

    @pytest.mark.unittest
    def test_select_within_bbox(self, borehole_collection):
        selected = borehole_collection.select_within_bbox(1.5, 1.5, 3.5, 5)
        assert all(selected.header["nr"].unique() == ["A", "D"])
        assert all(selected.data["nr"].unique() == ["A", "D"])

    @pytest.mark.unittest
    def test_select_with_points(self, borehole_collection):
        selection_points = [Point(1, 1), Point(4, 4), Point(1, 4), Point(4, 1)]
        selection_gdf = gpd.GeoDataFrame(
            {"id": [0, 1, 2, 3]}, geometry=selection_points
        )
        selection = borehole_collection.select_with_points(selection_gdf, 1.1)
        assert selection.n_points == 3

        selection_inverted = borehole_collection.select_with_points(
            selection_gdf, 1.1, invert=True
        )
        assert selection_inverted.n_points == 2

    @pytest.mark.unittest
    def test_select_with_lines(self, borehole_collection):
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame({"id": [0, 1]}, geometry=selection_lines)
        selection = borehole_collection.select_with_lines(selection_gdf, 0.6)
        assert selection.n_points == 2
        selection_inverted = borehole_collection.select_with_lines(
            selection_gdf, 0.6, invert=True
        )
        assert selection_inverted.n_points == 3

    @pytest.mark.unittest
    def test_select_within_polygons(self, borehole_collection):
        selection_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 3)))]
        selection_gdf = gpd.GeoDataFrame({"id": [0]}, geometry=selection_polygon)

        selection = borehole_collection.select_within_polygons(selection_gdf)
        assert selection.n_points == 1

        selection_inverted = borehole_collection.select_within_polygons(
            selection_gdf, invert=True
        )
        assert selection_inverted.n_points == 4

        selection_buffer = borehole_collection.select_within_polygons(
            selection_gdf, buffer=0.7
        )
        assert selection_buffer.n_points == 2

        selection_inverted_buffer = borehole_collection.select_within_polygons(
            selection_gdf, buffer=0.7, invert=True
        )
        assert selection_inverted_buffer.n_points == 3

    @pytest.mark.unittest
    def test_select_by_depth(self, borehole_collection):
        assert borehole_collection.select_by_depth(top_min=0).n_points == 4
        assert borehole_collection.select_by_depth(top_max=0).n_points == 1
        assert borehole_collection.select_by_depth(end_min=-3.5).n_points == 2
        assert (
            borehole_collection.select_by_depth(
                top_min=0, top_max=0.3, end_max=-4
            ).n_points
            == 2
        )

    @pytest.mark.unittest
    def test_select_by_length(self, borehole_collection):
        assert borehole_collection.select_by_length(min_length=4).n_points == 2
        assert borehole_collection.select_by_length(max_length=4).n_points == 3
        assert (
            borehole_collection.select_by_length(min_length=3, max_length=5).n_points
            == 4
        )

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_collection):
        selected = borehole_collection.select_by_values("lith", ["V", "K"], how="or")

        expected_nrs = ["A", "B", "C", "D"]
        selected_nrs = selected.data["nr"].unique()

        assert_array_equal(selected_nrs, expected_nrs)

        selected = borehole_collection.select_by_values("lith", ["V", "K"], how="and")

        expected_nrs = ["B", "D"]
        selected_nrs = selected.data["nr"].unique()

        assert_array_equal(selected_nrs, expected_nrs)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, borehole_collection):
        # Test slicing with respect to depth below the surface.
        upper, lower = 0.6, 2.4
        sliced = borehole_collection.slice_depth_interval(upper, lower)

        layers_per_borehole = sliced.data["nr"].value_counts()
        expected_layer_count = [3, 3, 3, 3, 2]

        assert len(sliced.data) == 14
        assert sliced.n_points == 5
        assert sliced.data["top"].min() == upper
        assert sliced.data["bottom"].max() == lower
        assert_array_equal(layers_per_borehole, expected_layer_count)

        # Test slicing without updating layer boundaries.
        sliced = borehole_collection.slice_depth_interval(
            upper, lower, update_layer_boundaries=False
        )

        expected_tops_of_slice = [0.0, 0.6, 0.0, 0.5, 0.5]
        expected_bottoms_of_slice = [2.5, 2.5, 2.9, 2.5, 2.5]

        tops_of_slice = sliced.data.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.data.groupby("nr")["bottom"].max()

        assert len(sliced.data) == 14
        assert sliced.n_points == 5
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        # Test slicing with respect to a vertical reference plane.
        nap_upper, nap_lower = -2, -3
        sliced = borehole_collection.slice_depth_interval(
            nap_upper, nap_lower, relative_to_vertical_reference=True
        )

        expected_tops_of_slice = [2.2, 2.3, 2.25, 2.1, 1.9]
        expected_bottoms_of_slice = [3.2, 3.3, 3.25, 3.0, 2.9]

        tops_of_slice = sliced.data.groupby("nr")["top"].min()
        bottoms_of_slice = sliced.data.groupby("nr")["bottom"].max()

        assert len(sliced.data) == 11
        assert sliced.n_points == 5
        assert_array_equal(tops_of_slice, expected_tops_of_slice)
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

        # Test slices that return empty objects.
        empty_slice = borehole_collection.slice_depth_interval(-2, -1)
        empty_slice_nap = borehole_collection.slice_depth_interval(
            3, 2, relative_to_vertical_reference=True
        )

        assert len(empty_slice.data) == 0
        assert len(empty_slice_nap.data) == 0
        assert empty_slice.n_points == 0
        assert empty_slice_nap.n_points == 0

        # Test slicing using only an upper boundary or lower boundary.
        upper = 4
        sliced = borehole_collection.slice_depth_interval(upper)

        expected_boreholes = ["A", "C"]

        assert len(sliced.data) == 2
        assert sliced.n_points == 2
        assert_array_equal(sliced.data["nr"], expected_boreholes)

        nap_lower = -0.5
        sliced = borehole_collection.slice_depth_interval(
            lower_boundary=nap_lower, relative_to_vertical_reference=True
        )

        bottoms_of_slice = sliced.data.groupby("nr")["bottom"].max()
        expected_bottoms_of_slice = [0.7, 0.8, 0.75, 0.6, 0.4]

        assert len(sliced.data) == 7
        assert sliced.n_points == 5
        assert_array_equal(bottoms_of_slice, expected_bottoms_of_slice)

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_collection):
        sliced = borehole_collection.slice_by_values("lith", "Z")

        expected_boreholes_with_sand = ["A", "C", "D", "E"]
        expected_length = 10

        assert_array_equal(sliced.data["nr"].unique(), expected_boreholes_with_sand)
        assert np.all(sliced.data["lith"] == "Z")
        assert len(sliced.data) == expected_length
        assert sliced.n_points == 4

        sliced = borehole_collection.slice_by_values("lith", "Z", invert=True)

        expected_boreholes_without_sand = ["A", "B", "C", "D"]
        expected_length = 15

        assert_array_equal(sliced.data["nr"].unique(), expected_boreholes_without_sand)
        assert ~np.any(sliced.data["lith"] == "Z")
        assert len(sliced.data) == expected_length
        assert sliced.n_points == 4

    @pytest.mark.unittest
    def test_select_by_condition(self, borehole_collection):
        # same selection with this method is used in test_data_objects
        selected = borehole_collection.select_by_condition(
            borehole_collection.data["lith"] == "V"
        )
        assert isinstance(selected, BoreholeCollection)
        assert selected.n_points == 2

    @pytest.mark.unittest
    def test_get_area_labels(self, borehole_collection):
        label_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 4)))]
        label_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=label_polygon, crs=28992)
        # Return variant
        output = borehole_collection.get_area_labels(label_gdf, "id")
        assert_almost_equal(output["id"].sum(), 2)
        # In-place variant
        borehole_collection.get_area_labels(label_gdf, "id", include_in_header=True)
        assert_almost_equal(borehole_collection.header["id"].sum(), 2)

    @pytest.mark.unittest
    def test_get_area_labels_multiple(self, borehole_collection):
        label_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 4)))]
        label_gdf = gpd.GeoDataFrame(
            {"id": [1], "col2": ["string_data"]}, geometry=label_polygon, crs=28992
        )
        # Return variant
        output = borehole_collection.get_area_labels(label_gdf, ["id", "col2"])
        assert_almost_equal(output["id"].sum(), 2)
        assert_equal(output["col2"].value_counts()["string_data"], 2)
        # In-place variant
        borehole_collection.get_area_labels(
            label_gdf, ["id", "col2"], include_in_header=True
        )
        assert_almost_equal(borehole_collection.header["id"].sum(), 2)
        assert_equal(
            borehole_collection.header["col2"].value_counts()["string_data"], 2
        )

    @pytest.mark.unittest
    def test_to_datafusiontools(self, borehole_collection):
        # More detailed tests are in TestLayeredData in test_data_objects.py
        dft = borehole_collection.to_datafusiontools("lith")
        assert np.all([isinstance(d, geodataclass.Data) for d in dft])

        outfile = Path("dft.pkl")
        borehole_collection.to_datafusiontools("lith", outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_qgis3d(self, borehole_collection):
        outfile = Path("temp.gpkg")
        borehole_collection.to_qgis3d(outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_geoparquet(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.geoparquet"
        borehole_collection.to_geoparquet(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_shape(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.shp"
        borehole_collection.to_shape(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_geopackage(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.gpkg"
        borehole_collection.to_geopackage(outfile)
        assert outfile.is_file()
        layers = gpd.list_layers(outfile)
        assert_array_equal(layers["name"], ["header", "data"])
        assert_array_equal(layers["geometry_type"], ["Point", None])

    @pytest.mark.unittest
    def test_to_parquet(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.gpkg"
        borehole_collection.to_parquet(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_csv(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.gpkg"
        borehole_collection.to_csv(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_pickle(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.pkl"
        borehole_collection.to_pickle(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_pyvista_cylinders(self, borehole_collection):
        # More detailed tests are in TestLayeredData in test_data_objects.py
        cylinders = borehole_collection.to_pyvista_cylinders("lith", radius=0.1)
        assert isinstance(cylinders, pv.MultiBlock)

    @pytest.mark.unittest
    def test_to_pyvista_grid(self, borehole_collection):
        # More detailed tests are in TestLayeredData in test_data_objects.py
        grid = borehole_collection.to_pyvista_grid("lith")
        assert isinstance(grid, pv.UnstructuredGrid)

    # @pytest.mark.integrationtest
    # def test_surface_level_update(self, borehole_collection, update_raster):
    #     borehole_collection.update_surface_level_from_raster(update_raster, how="replace")
    #     print("stop")


class TestBoreholeCollection:
    @pytest.mark.unittest
    def test_get_cumulative_thickness(self, borehole_collection):
        expected_sand_thickness = [2.2, 0.0, 2.6, 0.5, 3.0]

        # Test single lithology
        borehole_collection.get_cumulative_thickness(
            "lith", "Z", include_in_header=True
        )
        assert_almost_equal(
            borehole_collection.header["Z_thickness"], expected_sand_thickness
        )

        # Test multiple lithologies
        borehole_collection.get_cumulative_thickness(
            "lith", ["Z", "K"], include_in_header=True
        )
        expected_clay_thickness = [2.0, 2.0, 2.9, 1.1, 0.0]
        expected_sand_thickness = [2.2, 0.0, 2.6, 0.5, 3.0]

        assert_almost_equal(
            borehole_collection.header["K_thickness"], expected_clay_thickness
        )
        assert_almost_equal(
            borehole_collection.header["Z_thickness"], expected_sand_thickness
        )

    @pytest.mark.unittest
    def test_get_layer_top(self, borehole_collection):
        borehole_collection.get_layer_top("lith", ["Z", "K"], include_in_header=True)

        expected_sand_top = [1.5, np.nan, 2.9, 2.5, 0.0]
        expected_clay_top = [0.0, 0.0, 0.0, 0.0, np.nan]

        assert_almost_equal(borehole_collection.header["K_top"], expected_clay_top)
        assert_almost_equal(borehole_collection.header["Z_top"], expected_sand_top)

    @pytest.mark.unittest
    def test_to_kingdom(self, borehole_collection):
        outfile = Path("temp_kingdom.csv")
        tdfile = Path(outfile.parent, f"{outfile.stem}_TDCHART{outfile.suffix}")
        borehole_collection.to_kingdom(outfile)
        assert outfile.is_file()
        assert tdfile.is_file()
        outfile.unlink()
        tdfile.unlink()


class TestCptCollection:
    @pytest.mark.unittest
    def test_slice_depth_interval(self, cpt_collection):
        upper, lower = 0.6, 4.4
        sliced = cpt_collection.slice_depth_interval(upper, lower)

        assert len(sliced.data) == 8
        assert sliced.n_points == 2
        assert sliced.data["depth"].min() >= upper
        assert sliced.data["depth"].max() <= lower

        upper, lower = 1.9, 0.9  # Elevations in NAP
        sliced = cpt_collection.slice_depth_interval(
            upper, lower, relative_to_vertical_reference=True
        )
        assert sliced.n_points == 1
        assert len(sliced.data) == 1
