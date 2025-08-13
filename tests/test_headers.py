import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
from shapely.geometry import LineString, Point, Polygon

from geost.base import LineHeader, PointHeader


class TestHeaders:
    @pytest.mark.unittest
    def test_init_header(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        assert "PointHeader instance containing 25 objects" in point_header.__repr__()

    @pytest.mark.unittest
    def test_single_get(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        point_header_sel = point_header.get("nr10")
        assert point_header_sel["nr"].iloc[0] == "nr10"

    @pytest.mark.unittest
    def test_multi_get(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        query = ["nr1", "nr5", "nr10", "nr15", "nr20"]
        point_header_sel = point_header.get(query)
        assert all([nr in point_header_sel["nr"].values for nr in query])

    @pytest.mark.unittest
    def test_change_horizontal_reference(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        point_header.change_horizontal_reference(32631)
        assert point_header.horizontal_reference == 32631
        assert_almost_equal(point_header["x"][0], 523402.3476207458)
        assert_almost_equal(point_header["y"][0], 5313544.160440822)

    @pytest.mark.unittest
    def test_change_vertical_reference(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        current_NAP_levels = point_header["surface"].copy()  # noqa: N806
        current_NAP_ends = point_header["end"].copy()  # noqa: N806
        # The offset between NAP (epsg:5709) and TAW (epsg:5710) is 2.28234 m.
        expected_TAW_levels = current_NAP_levels + 2.28234  # noqa: N806
        expected_TAW_ends = current_NAP_ends + 2.28234  # noqa: N806
        point_header.change_vertical_reference(5710)
        assert point_header.vertical_reference == 5710
        assert_allclose(point_header["surface"], expected_TAW_levels, atol=10e-4)
        assert_allclose(point_header["end"], expected_TAW_ends, atol=10e-4)

    @pytest.mark.unittest
    def test_select_within_bbox(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        point_header_sel = point_header.select_within_bbox(1, 1, 3, 3)
        point_header_sel_inverted = point_header.select_within_bbox(
            1, 1, 3, 3, invert=True
        )
        assert len(point_header_sel) == 9
        assert len(point_header_sel_inverted) == 16

    @pytest.mark.unittest
    def test_select_with_points(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        selection_points = [Point(1, 1), Point(4, 4), Point(1, 4), Point(4, 1)]
        selection_gdf = gpd.GeoDataFrame(
            {"id": [0, 1, 2, 3]}, geometry=selection_points
        )
        selected = point_header.select_with_points(selection_gdf, 1.1)
        assert len(selected) == 16

        inverted_selection = point_header.select_with_points(
            selection_gdf, 1.1, invert=True
        )
        assert len(inverted_selection) == 9

        # Select with Shapely object
        selected = point_header.select_with_points(selection_points[0], 1.1)
        assert len(selected) == 3

        # Select with iterable
        selected = point_header.select_with_points(selection_points, 1.1)
        assert len(selected) == 16

    @pytest.mark.unittest
    def test_select_with_lines(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame({"id": [0, 1]}, geometry=selection_lines)

        selected = point_header.select_with_lines(selection_gdf, 1)
        assert len(selected) == 21

        inverted_selection = point_header.select_with_lines(
            selection_gdf, 1, invert=True
        )
        assert len(inverted_selection) == 4

        # Select with Shapely object
        selected = point_header.select_with_lines(selection_lines[0], 1)
        assert len(selected) == 13

        # Select with iterable
        selected = point_header.select_with_lines(selection_lines, 1)
        assert len(selected) == 21

    @pytest.mark.unittest
    def test_select_within_polygons(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        selection_polygon = Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))
        selection_gdf = gpd.GeoDataFrame({"id": [0]}, geometry=[selection_polygon])

        # Geodataframe based selection
        selected = point_header.select_within_polygons(selection_gdf)
        assert len(selected) == 3

        selected_inverted = point_header.select_within_polygons(
            selection_gdf, invert=True
        )
        assert len(selected_inverted) == 22

        selected_buffered = point_header.select_within_polygons(
            selection_gdf, buffer=0.1
        )
        assert len(selected_buffered) == 11

        selected_inverted_buffered = point_header.select_within_polygons(
            selection_gdf, buffer=0.1, invert=True
        )
        assert len(selected_inverted_buffered) == 14

        # Shapely Polygon based selection
        selected = point_header.select_within_polygons(selection_polygon)
        assert len(selected) == 3
        assert_array_equal(selected["nr"], ["nr7", "nr13", "nr19"])

        # Selection with Iterable
        selected = point_header.select_within_polygons([selection_polygon])
        assert len(selected) == 3

    @pytest.mark.unittest
    def test_select_by_depth(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        assert len(point_header.select_by_depth(top_min=10).gdf) == 16
        assert len(point_header.select_by_depth(top_max=10).gdf) == 10
        assert len(point_header.select_by_depth(end_min=-10).gdf) == 10
        assert len(point_header.select_by_depth(end_max=-10).gdf) == 16
        assert len(point_header.select_by_depth(top_min=10, top_max=15).gdf) == 6
        assert (
            len(point_header.select_by_depth(top_min=10, top_max=15, end_max=-12).gdf)
            == 4
        )

    @pytest.mark.unittest
    def test_select_by_length(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        assert len(point_header.select_by_length(min_length=10)) == 21
        assert len(point_header.select_by_length(max_length=30)) == 15
        assert len(point_header.select_by_length(min_length=10, max_length=30)) == 11

    @pytest.mark.unittest
    def test_get_area_labels(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        label_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))]
        label_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=label_polygon, crs=28992)
        # Return variant
        output = point_header.get_area_labels(label_gdf, "id")
        assert_almost_equal(output["id"].sum(), 11)
        # In-place variant
        point_header.get_area_labels(label_gdf, "id", include_in_header=True)
        assert_almost_equal(point_header["id"].sum(), 11)

    @pytest.mark.unittest
    def test_get_area_labels_multiple(self, point_header_gdf):
        point_header = PointHeader(point_header_gdf, "NAP")
        label_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))]
        label_gdf = gpd.GeoDataFrame(
            {"id": [1], "col2": ["string_data"]}, geometry=label_polygon, crs=28992
        )
        # Return variant
        output = point_header.get_area_labels(label_gdf, ["id", "col2"])
        assert_almost_equal(output["id"].sum(), 11)
        assert_equal(output["col2"].value_counts()["string_data"], 11)
        # In-place variant
        point_header.get_area_labels(label_gdf, ("id", "col2"), include_in_header=True)
        assert_almost_equal(point_header["id"].sum(), 11)
        assert_equal(point_header["col2"].value_counts()["string_data"], 11)

    @pytest.mark.unittest
    def test_export_to_csv_mixin(self, point_header_gdf, tmp_path):
        point_header = PointHeader(point_header_gdf, "NAP")
        temp_file = tmp_path / r"pointheader.csv"
        point_header.to_csv(temp_file)
        assert temp_file.is_file()

    @pytest.mark.unittest
    def test_export_to_parquet_mixin(self, point_header_gdf, tmp_path):
        point_header = PointHeader(point_header_gdf, "NAP")
        temp_file = tmp_path / r"pointheader.parquet"
        point_header.to_parquet(temp_file)
        assert temp_file.is_file()

    @pytest.mark.unittest
    def test_export_to_shapefile_mixin(self, point_header_gdf, tmp_path):
        point_header = PointHeader(point_header_gdf, "NAP")
        temp_file = tmp_path / r"pointheader.shp"
        point_header.to_shape(temp_file)
        assert temp_file.is_file()

    @pytest.mark.unittest
    def test_export_to_geopackage_mixin(self, point_header_gdf, tmp_path):
        point_header = PointHeader(point_header_gdf, "NAP")
        temp_file = tmp_path / r"pointheader.gpkg"
        point_header.to_geopackage(temp_file)
        assert temp_file.is_file()

    @pytest.mark.unittest
    def test_export_to_geoparquet_mixin(self, point_header_gdf, tmp_path):
        point_header = PointHeader(point_header_gdf, "NAP")
        temp_file = tmp_path / r"pointheader.geoparquet"
        point_header.to_geoparquet(temp_file)
        assert temp_file.is_file()
