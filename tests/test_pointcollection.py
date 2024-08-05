from pathlib import Path

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

from geost import read_nlog_cores, read_sst_cores
from geost.new_base import BoreholeCollection, LayeredData


class TestPointCollection:
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
        borehole_collection_single_selection = borehole_collection.get("HB-8")
        assert borehole_collection_single_selection.header.gdf.iloc[0, 0] == "HB-8"

    @pytest.mark.unittest
    def test_get_multiple_objects(self, borehole_collection):
        borehole_collection_multi_selection = borehole_collection.get(["HB-8", "HB-6"])
        assert list(borehole_collection_multi_selection.header["nr"].unique()) == [
            "HB-6",
            "HB-8",
        ]

    @pytest.mark.unittest
    def test_add_header_column_to_data(self, borehole_collection):
        borehole_collection.header["test_data"] = [
            i for i in range(len(borehole_collection.header))
        ]
        borehole_collection.add_header_column_to_data("test_data")

        assert_allclose(borehole_collection.get("HB-6").data["test_data"], 0)
        assert_allclose(borehole_collection.get("HB-03").data["test_data"], 12)

    @pytest.mark.unittest
    def test_change_vertical_reference(self, borehole_data):
        borehole_collection_ok = LayeredData(borehole_data).to_collection()
        assert borehole_collection_ok.vertical_reference == 5709
        borehole_collection_ok.change_vertical_reference("Ostend height")
        assert borehole_collection_ok.vertical_reference == 5710

    @pytest.mark.unittest
    def test_change_horizontal_reference(self, borehole_data):
        borehole_collection_ok = LayeredData(borehole_data).to_collection()
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
    def test_slice_depth_interval(self, borehole_collection):
        # test with NAP reference
        slice1 = borehole_collection.slice_depth_interval(
            lower_boundary=-3, upper_boundary=0
        )

        assert len(slice1.data) == 15
        assert len(slice1.header) == 6

        # test with depth reference
        slice2 = borehole_collection.slice_depth_interval(
            lower_boundary=10, upper_boundary=5, vertical_reference="depth"
        )

        assert len(slice2.data) == 6
        assert len(slice2.header) == 4
        # vertical_reference of slice must be as specified in function
        assert slice2.vertical_reference == "depth"
        # original vertical_reference must be kept the same as before function call
        assert borehole_collection.vertical_reference == "NAP"

        # test with surfacelevel reference and when a reference is specified but same as original
        borehole_collection.change_vertical_reference("surfacelevel")
        slice3 = borehole_collection.slice_depth_interval(
            upper_boundary=-5, lower_boundary=-10, vertical_reference="surfacelevel"
        )

        assert len(slice3.data) == 6
        assert len(slice3.header) == 4

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_collection):
        layers_k = borehole_collection.slice_by_values("lith", "K")
        layers_ks2 = borehole_collection.slice_by_values("lith_comb", "Ks2")
        layers_h2 = borehole_collection.slice_by_values("org", "H2")
        layers_v_z = borehole_collection.slice_by_values("lith", ["V", "Z"])

        assert len(layers_k.data) == 188
        assert len(layers_k.header) == 13
        assert len(layers_ks2.data) == 19
        assert len(layers_ks2.header) == 6
        assert len(layers_h2.data) == 3
        assert len(layers_h2.header) == 3
        assert len(layers_v_z.data) == 23
        assert len(layers_v_z.header) == 11

    @pytest.mark.unittest
    def test_inverted_slice_by_values(self, borehole_collection):
        layers_non_k = borehole_collection.slice_by_values("lith", "K", invert=True)
        layers_non_ks2 = borehole_collection.slice_by_values(
            "lith_comb", "Ks2", invert=True
        )
        layers_non_h2 = borehole_collection.slice_by_values("org", "H2", invert=True)
        layers_non_v_z = borehole_collection.slice_by_values(
            "lith", ["V", "Z"], invert=True
        )

        assert len(layers_non_k.data) == 162
        assert len(layers_non_k.header) == 13
        assert len(layers_non_ks2.data) == 331
        assert len(layers_non_ks2.header) == 13
        assert len(layers_non_h2.data) == 347
        assert len(layers_non_h2.header) == 13
        assert len(layers_non_v_z.data) == 327
        assert len(layers_non_v_z.header) == 13

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
