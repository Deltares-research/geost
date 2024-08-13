from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray as rio
import xarray as xr
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_equal,
)
from shapely.geometry import LineString, Point, Polygon

from geost import read_borehole_table, read_nlog_cores
from geost.new_base import BoreholeCollection, LayeredData


class TestCollection:
    nlog_stratstelsel_parquet = (
        Path(__file__).parent / "data/test_nlog_stratstelsel_20230807.parquet"
    )

    @pytest.fixture
    def borehole_df_ok(self):
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
                "surface": mv,
                "end": end,
                "top": top,
                "bottom": bottom,
                "data_string": data_string,
                "data_int": data_int,
                "data_float": data_float,
            }
        )

        return dataframe

    @pytest.fixture
    def borehole_df_bad_validation(self):
        nr = np.full(10, 1)
        x = np.full(10, 139370)
        y = np.full(10, 455540)
        mv = np.full(10, 1.0)
        end = np.full(10, 4.0)
        top = np.array([1, 0.5, 0, -0.5, -1.5, -2, -2.5, -3, -3.2, -4.2])
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
                "surface": mv,
                "end": end,
                "top": top,
                "bottom": bottom,
                "data_string": data_string,
                "data_int": data_int,
                "data_float": data_float,
            }
        )

        return dataframe

    @pytest.fixture
    def header_missing_object(self):
        dataframe = pd.DataFrame(
            {
                "nr": ["B-02"],
                "x": [100000],
                "y": [400000],
                "surface": [0],
                "end": [-8],
            }
        )

        return dataframe

    @pytest.fixture
    def header_surplus_objects(self):
        dataframe = pd.DataFrame(
            {
                "nr": ["B-01", "B-02", "B-03"],
                "x": [139370, 100000, 110000],
                "y": [455540, 400000, 410000],
                "surface": [1, 0, -1],
                "end": [-4, -8, -9],
            }
        )

        return dataframe

    @pytest.fixture
    def update_raster(self):
        x_coors = np.arange(127000, 128500, 500)
        y_coors = np.arange(503000, 501500, -500)
        data = np.ones((3, 3))
        array = xr.DataArray(data, {"x": x_coors, "y": y_coors})
        return array

    @pytest.mark.unittest
    def test_get_single_object(self, borehole_collection):
        borehole_collection_single_selection = borehole_collection.get("A")
        assert borehole_collection_single_selection.header.gdf.iloc[0, 0] == "A"

    @pytest.mark.unittest
    def test_get_multiple_objects(self, borehole_collection):
        borehole_collection_multi_selection = borehole_collection.get(["A", "B"])
        assert list(borehole_collection_multi_selection.header["nr"].unique()) == [
            "A",
            "B",
        ]

    @pytest.mark.unittest
    def test_add_header_column_to_data(self, borehole_collection):
        borehole_collection.header["test_data"] = [
            i for i in range(len(borehole_collection.header))
        ]
        borehole_collection.add_header_column_to_data("test_data")

        assert_allclose(borehole_collection.get("A").data["test_data"], 0)
        assert_allclose(borehole_collection.get("B").data["test_data"], 1)

    @pytest.mark.unittest
    def test_change_vertical_reference(self, borehole_data):
        borehole_collection_ok = borehole_data.to_collection()
        assert borehole_collection_ok.vertical_reference == 5709
        borehole_collection_ok.change_vertical_reference("Ostend height")
        assert borehole_collection_ok.vertical_reference == 5710

    @pytest.mark.unittest
    def test_change_horizontal_reference(self, borehole_data):
        borehole_collection_ok = borehole_data.to_collection()
        assert borehole_collection_ok.horizontal_reference == 28992
        borehole_collection_ok.change_horizontal_reference(32631)
        assert borehole_collection_ok.horizontal_reference == 32631
        assert_almost_equal(
            borehole_collection_ok.data["x"][0], borehole_collection_ok.header["x"][0]
        )
        assert_almost_equal(
            borehole_collection_ok.data["x"][0], 523403.281803, decimal=5
        )
        assert_almost_equal(
            borehole_collection_ok.data["y"][0], borehole_collection_ok.header["y"][0]
        )
        assert_almost_equal(
            borehole_collection_ok.data["y"][0], 5313546.187669, decimal=5
        )

    @pytest.mark.unittest
    def test_change_horizontal_reference_also_data_columns_inclined(
        self, nlog_borehole_collection
    ):
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
    def test_select_within_bbox(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
        borehole_collection_selected = borehole_collection.select_within_bbox(
            1.5, 3.5, 1.5, 5
        )
        # The selection results boreholes 'A' and 'D'
        assert all(borehole_collection_selected.header.gdf["nr"].unique() == ["A", "D"])
        assert all(borehole_collection_selected.data.df["nr"].unique() == ["A", "D"])

    @pytest.mark.unittest
    def test_select_with_points(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
        selection_points = [Point(1, 1), Point(4, 4), Point(1, 4), Point(4, 1)]
        selection_gdf = gpd.GeoDataFrame(
            {"id": [0, 1, 2, 3]}, geometry=selection_points
        )
        collection_sel = borehole_collection.select_with_points(selection_gdf, 1.1)
        collection_sel_inverted = borehole_collection.select_with_points(
            selection_gdf, 1.1, invert=True
        )
        assert collection_sel.n_points == 3
        assert collection_sel_inverted.n_points == 2

    @pytest.mark.unittest
    def test_select_with_lines(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame({"id": [0, 1]}, geometry=selection_lines)
        collection_sel = borehole_collection.select_with_lines(selection_gdf, 0.6)
        collection_sel_inverted = borehole_collection.select_with_lines(
            selection_gdf, 0.6, invert=True
        )
        assert collection_sel.n_points == 2
        assert collection_sel_inverted.n_points == 3

    @pytest.mark.unittest
    def test_select_within_polygons(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
        selection_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 3)))]
        selection_gdf = gpd.GeoDataFrame({"id": [0]}, geometry=selection_polygon)

        # Polygon selection without buffer
        collection_sel = borehole_collection.select_within_polygons(selection_gdf)
        collection_sel_inverted = borehole_collection.select_within_polygons(
            selection_gdf, invert=True
        )
        assert collection_sel.n_points == 1
        assert collection_sel_inverted.n_points == 4

        # Polygon selection with a buffer
        collection_sel_buff = borehole_collection.select_within_polygons(
            selection_gdf, buffer=0.7
        )
        collection_sel_inverted_buff = borehole_collection.select_within_polygons(
            selection_gdf, buffer=0.7, invert=True
        )
        assert collection_sel_buff.n_points == 2
        assert collection_sel_inverted_buff.n_points == 3

    @pytest.mark.unittest
    def test_select_by_depth(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
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
    def test_select_by_length(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
        assert borehole_collection.select_by_length(min_length=4).n_points == 2
        assert borehole_collection.select_by_length(max_length=4).n_points == 3
        assert (
            borehole_collection.select_by_length(min_length=3, max_length=5).n_points
            == 4
        )

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_data):
        borehole_collection = borehole_data.to_collection()
        assert borehole_collection.select_by_length(min_length=4).n_points == 2
        assert borehole_collection.select_by_length(max_length=4).n_points == 3
        assert (
            borehole_collection.select_by_length(min_length=3, max_length=5).n_points
            == 4
        )

    @pytest.mark.unittest
    def test_slice_depth_interval(self, borehole_collection):
        pass

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_collection):
        pass

    @pytest.mark.unittest
    def test_inverted_slice_by_values(self, borehole_collection):
        pass

    @pytest.mark.integrationtest
    def test_validation_pass(self, capfd, borehole_df_ok):
        BoreholeCollection(borehole_df_ok)
        out, err = capfd.readouterr()
        # Since no warning line was printed, the length of out must be 0
        assert_equal(len(out), 0)

    @pytest.mark.integrationtest
    def test_validation_fail(self, capfd, borehole_df_bad_validation):
        BoreholeCollection(borehole_df_bad_validation)
        out, err = capfd.readouterr()
        # Check if required warnings are printed. Note that changing warning messages
        # will make this test fail.
        assert 'but is required to be "stringlike type"' in out
        assert 'data in column "bottom" failed check "< top" for 1 rows: [1]' in out
        assert 'data in column "end" failed check "< mv" for 1 rows: [1]' in out

    @pytest.mark.integrationtest
    def test_header_mismatch(
        self, capfd, borehole_df_ok, header_missing_object, header_surplus_objects
    ):
        # Situation #1: More unique objects in data table than listed in header
        collection = BoreholeCollection(borehole_df_ok, header=header_missing_object)
        out, err = capfd.readouterr()
        assert "Header does not cover all unique objects in data" in out

        # Situation #2: More objects in header table than in data table
        collection = BoreholeCollection(borehole_df_ok, header=header_surplus_objects)
        out, err = capfd.readouterr()
        assert "Header covers more objects than present in the data table" in out

    # @pytest.mark.integrationtest
    # def test_surface_level_update(self, borehole_collection, update_raster):
    #     borehole_collection.update_surface_level_from_raster(update_raster, how="replace")
    #     print("stop")
