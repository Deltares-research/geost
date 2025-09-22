import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
from shapely.geometry import LineString, Polygon


class TestPointHeader:
    @pytest.mark.unittest
    def test_get(self, point_header):
        selected = point_header.gsthd.get("nr10")
        assert selected.headertype == point_header.headertype
        assert_array_equal(selected["nr"].iloc[0], "nr10")

        nrs = ["nr1", "nr5", "nr10", "nr15", "nr20"]
        selected = point_header.gsthd.get(nrs)
        assert_array_equal(selected["nr"].values, nrs)

    @pytest.mark.unittest
    def test_change_horizontal_reference(self, point_header):
        point_header.gsthd.change_horizontal_reference(32631)
        assert point_header.crs == 32631
        assert_almost_equal(point_header["x"][0], 523402.3476207458)
        assert_almost_equal(point_header["y"][0], 5313544.160440822)

    @pytest.mark.unittest
    def test_change_vertical_reference(self, point_header):
        NAP_TAW_offset = 2.28234  # noqa: N806
        expected_TAW_levels = point_header["surface"] + NAP_TAW_offset  # noqa: N806
        expected_TAW_ends = point_header["end"] + NAP_TAW_offset  # noqa: N806
        point_header.gsthd.change_vertical_reference(5709, 5710)
        assert_allclose(point_header["surface"], expected_TAW_levels, atol=10e-4)
        assert_allclose(point_header["end"], expected_TAW_ends, atol=10e-4)

    @pytest.mark.unittest
    def test_select_within_bbox(self, point_header):
        selected = point_header.gsthd.select_within_bbox(1, 1, 3, 3)
        assert selected.headertype == point_header.headertype
        assert len(selected) == 9

        # Test inverted selection: everything outside the bbox
        selected = point_header.gsthd.select_within_bbox(1, 1, 3, 3, invert=True)
        assert len(selected) == 16

    @pytest.mark.unittest
    def test_select_with_points(self, point_header):
        selection_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([1, 4, 1, 4], [1, 4, 4, 1]), crs=28992
        )
        buffer = 1.1
        selected = point_header.gsthd.select_with_points(selection_points, buffer)
        assert len(selected) == 16
        assert selected.headertype == point_header.headertype

        inverted_selection = point_header.gsthd.select_with_points(
            selection_points, buffer, invert=True
        )
        assert len(inverted_selection) == 9

        # Select with Shapely object
        selected = point_header.gsthd.select_with_points(
            selection_points["geometry"].iloc[0], buffer
        )
        assert len(selected) == 3

        # Select with iterable
        selected = point_header.gsthd.select_with_points(
            selection_points["geometry"].to_list(), buffer
        )
        assert len(selected) == 16

    @pytest.mark.unittest
    def test_select_with_lines(self, point_header):
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame(geometry=selection_lines, crs=28992)

        buffer = 1
        selected = point_header.gsthd.select_with_lines(selection_gdf, buffer)
        assert len(selected) == 21
        assert selected.headertype == point_header.headertype

        inverted_selection = point_header.gsthd.select_with_lines(
            selection_gdf, buffer, invert=True
        )
        assert len(inverted_selection) == 4

        # Select with Shapely object
        selected = point_header.gsthd.select_with_lines(selection_lines[0], buffer)
        assert len(selected) == 13

        # Select with iterable
        selected = point_header.gsthd.select_with_lines(selection_lines, buffer)
        assert len(selected) == 21

    @pytest.mark.unittest
    def test_select_within_polygons(self, point_header):
        selection_polygon = Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))
        selection_gdf = gpd.GeoDataFrame(geometry=[selection_polygon], crs=28992)

        # Geodataframe based selection
        selected = point_header.gsthd.select_within_polygons(selection_gdf)
        assert len(selected) == 3
        assert selected.headertype == point_header.headertype

        selected_inverted = point_header.gsthd.select_within_polygons(
            selection_gdf, invert=True
        )
        assert len(selected_inverted) == 22

        selected_buffered = point_header.gsthd.select_within_polygons(
            selection_gdf, buffer=0.1
        )
        assert len(selected_buffered) == 11

        selected_inverted_buffered = point_header.gsthd.select_within_polygons(
            selection_gdf, buffer=0.1, invert=True
        )
        assert len(selected_inverted_buffered) == 14

        # Shapely Polygon based selection
        selected = point_header.gsthd.select_within_polygons(selection_polygon)
        assert len(selected) == 3
        assert_array_equal(selected["nr"], ["nr7", "nr13", "nr19"])

        # Selection with Iterable
        selected = point_header.gsthd.select_within_polygons([selection_polygon])
        assert len(selected) == 3

    @pytest.mark.unittest
    def test_select_by_depth(self, point_header):
        selected = point_header.gsthd.select_by_depth(top_min=10)
        assert len(selected) == 16
        assert selected.headertype == point_header.headertype

        assert len(point_header.gsthd.select_by_depth(top_max=10)) == 10
        assert len(point_header.gsthd.select_by_depth(end_min=-10)) == 10
        assert len(point_header.gsthd.select_by_depth(end_max=-10)) == 16
        assert len(point_header.gsthd.select_by_depth(top_min=10, top_max=15)) == 6
        assert (
            len(point_header.gsthd.select_by_depth(top_min=10, top_max=15, end_max=-12))
            == 4
        )

    @pytest.mark.unittest
    def test_select_by_length(self, point_header):
        selected = point_header.gsthd.select_by_length(min_length=10)
        assert selected.headertype == point_header.headertype
        assert len(selected) == 21
        assert len(point_header.gsthd.select_by_length(max_length=30)) == 15
        assert (
            len(point_header.gsthd.select_by_length(min_length=10, max_length=30)) == 11
        )

    @pytest.mark.unittest
    def test_get_area_labels(self, point_header):
        label_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))]
        label_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=label_polygon, crs=28992)

        expected_output = pd.DataFrame(
            {
                "nr": [
                    "nr2",  # These numbers get a label assigned, the rest is <NA>
                    "nr6",
                    "nr7",
                    "nr8",
                    "nr12",
                    "nr13",
                    "nr14",
                    "nr18",
                    "nr19",
                    "nr20",
                    "nr24",
                ],
                "id": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=[1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23],
        )
        # Return variant
        output = point_header.gsthd.get_area_labels(label_gdf, "id")
        pd.testing.assert_frame_equal(output.dropna(subset="id"), expected_output)
        # In-place variant
        point_header.gsthd.get_area_labels(label_gdf, "id", include_in_header=True)
        assert_almost_equal(point_header["id"].sum(), 11)

    @pytest.mark.unittest
    def test_get_area_labels_multiple(self, point_header):
        label_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 2)))]
        label_gdf = gpd.GeoDataFrame(
            {"id": [1], "col2": ["string_data"]}, geometry=label_polygon, crs=28992
        )
        # Return variant
        output = point_header.gsthd.get_area_labels(label_gdf, ["id", "col2"])
        assert_almost_equal(output["id"].sum(), 11)
        assert_equal(output["col2"].value_counts()["string_data"], 11)
        # In-place variant
        point_header.gsthd.get_area_labels(
            label_gdf, ["id", "col2"], include_in_header=True
        )
        assert_almost_equal(point_header["id"].sum(), 11)
        assert_equal(point_header["col2"].value_counts()["string_data"], 11)
