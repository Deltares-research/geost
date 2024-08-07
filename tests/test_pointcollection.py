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
from geost.borehole import BoreholeCollection


class TestPointCollection:
    nlog_stratstelsel_parquet = (
        Path(__file__).parent / "data/test_nlog_stratstelsel_20230807.parquet"
    )
    borehole_file = Path(__file__).parent / "data" / "test_boreholes.parquet"

    @pytest.fixture
    def boreholes(self):
        borehole_collection = read_sst_cores(self.borehole_file)
        return borehole_collection

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
                "mv": mv,
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
                "mv": mv,
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
                "mv": [0],
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
                "mv": [1, 0, -1],
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
    def test_get_single_object(self, boreholes):
        borehole_collection_single_selection = boreholes.get("HB-8")
        assert borehole_collection_single_selection.header.iloc[0, 0] == "HB-8"

    @pytest.mark.unittest
    def test_get_multiple_objects(self, boreholes):
        borehole_collection_multi_selection = boreholes.get(["HB-8", "HB-6"])
        assert list(borehole_collection_multi_selection.header["nr"].unique()) == [
            "HB-6",
            "HB-8",
        ]

    @pytest.mark.unittest
    def test_change_vertical_reference(self, borehole_df_ok):
        borehole_collection_ok = BoreholeCollection(borehole_df_ok)
        assert borehole_collection_ok.vertical_reference == "NAP"

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
        borehole_collection_ok.change_vertical_reference("surfacelevel")
        assert_array_almost_equal(
            borehole_collection_ok.data["top"], target_top_surfacelevel
        )
        assert_array_almost_equal(
            borehole_collection_ok.data["bottom"], target_bottom_surfacelevel
        )

        # From surfacelevel to NAP
        borehole_collection_ok.change_vertical_reference("NAP")
        assert_array_almost_equal(borehole_collection_ok.data["top"], target_top_nap)
        assert_array_almost_equal(
            borehole_collection_ok.data["bottom"], target_bottom_nap
        )

        # From NAP to depth
        borehole_collection_ok.change_vertical_reference("depth")
        assert_array_almost_equal(borehole_collection_ok.data["top"], target_top_depth)
        assert_array_almost_equal(
            borehole_collection_ok.data["bottom"], target_bottom_depth
        )

        # From depth to NAP
        borehole_collection_ok.change_vertical_reference("NAP")
        assert_array_almost_equal(borehole_collection_ok.data["top"], target_top_nap)
        assert_array_almost_equal(
            borehole_collection_ok.data["bottom"], target_bottom_nap
        )

        # From surfacelevel to depth
        borehole_collection_ok.change_vertical_reference("surfacelevel")
        borehole_collection_ok.change_vertical_reference("depth")
        assert_array_almost_equal(borehole_collection_ok.data["top"], target_top_depth)
        assert_array_almost_equal(
            borehole_collection_ok.data["bottom"], target_bottom_depth
        )

        # From depth to surfacelevel
        borehole_collection_ok.change_vertical_reference("surfacelevel")
        assert_array_almost_equal(
            borehole_collection_ok.data["top"], target_top_surfacelevel
        )
        assert_array_almost_equal(
            borehole_collection_ok.data["bottom"], target_bottom_surfacelevel
        )

    @pytest.mark.unittest
    def test_change_horizontal_reference_only_geometry(self, borehole_df_ok):
        borehole_collection_ok = BoreholeCollection(borehole_df_ok)
        assert_equal(borehole_collection_ok.horizontal_reference, 28992)
        borehole_collection_ok.change_horizontal_reference(32631, only_geometries=True)
        assert_equal(borehole_collection_ok.header.crs.name, "WGS 84 / UTM zone 31N")
        assert_equal(borehole_collection_ok.horizontal_reference, 32631)

    @pytest.mark.unittest
    def test_change_horizontal_reference_also_data_columns(self, borehole_df_ok):
        borehole_collection_ok = BoreholeCollection(borehole_df_ok)
        assert_equal(borehole_collection_ok.horizontal_reference, 28992)
        borehole_collection_ok.change_horizontal_reference(32631, only_geometries=False)
        assert_equal(borehole_collection_ok.header.crs.name, "WGS 84 / UTM zone 31N")
        assert_equal(borehole_collection_ok.horizontal_reference, 32631)
        assert_almost_equal(borehole_collection_ok.header.x[0], 647927.91, decimal=2)
        assert_almost_equal(borehole_collection_ok.header.y[0], 5.773014e6, decimal=0)
        assert_almost_equal(borehole_collection_ok.data.x[0], 647927.91, decimal=2)
        assert_almost_equal(borehole_collection_ok.data.y[0], 5.773014e6, decimal=0)

    @pytest.mark.unittest
    def test_change_horizontal_reference_also_data_columns_inclined(self):
        borehole_collection_ok = read_nlog_cores(self.nlog_stratstelsel_parquet)
        assert_equal(borehole_collection_ok.horizontal_reference, 28992)
        borehole_collection_ok.change_horizontal_reference(32631, only_geometries=False)
        assert_equal(borehole_collection_ok.header.crs.name, "WGS 84 / UTM zone 31N")
        assert_equal(borehole_collection_ok.horizontal_reference, 32631)
        assert_almost_equal(borehole_collection_ok.header.x[0], 532641.76, decimal=2)
        assert_almost_equal(borehole_collection_ok.header.y[0], 6.170701e6, decimal=0)
        assert_almost_equal(borehole_collection_ok.data.x_bot[0], 532650.77, decimal=2)
        assert_almost_equal(borehole_collection_ok.data.y_bot[0], 6.170700e6, decimal=0)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, boreholes):
        # test with NAP reference
        slice1 = boreholes.slice_depth_interval(lower_boundary=-3, upper_boundary=0)

        assert len(slice1.data) == 15
        assert len(slice1.header) == 6

        # test with depth reference
        slice2 = boreholes.slice_depth_interval(
            lower_boundary=10, upper_boundary=5, vertical_reference="depth"
        )

        assert len(slice2.data) == 6
        assert len(slice2.header) == 4
        # vertical_reference of slice must be as specified in function
        assert slice2.vertical_reference == "depth"
        # original vertical_reference must be kept the same as before function call
        assert boreholes.vertical_reference == "NAP"

        # test with surfacelevel reference and when a reference is specified but same as original
        boreholes.change_vertical_reference("surfacelevel")
        slice3 = boreholes.slice_depth_interval(
            upper_boundary=-5, lower_boundary=-10, vertical_reference="surfacelevel"
        )

        assert len(slice3.data) == 6
        assert len(slice3.header) == 4

    @pytest.mark.unittest
    def test_slice_by_values(self, boreholes):
        layers_k = boreholes.slice_by_values("lith", "K")
        layers_ks2 = boreholes.slice_by_values("lith_comb", "Ks2")
        layers_h2 = boreholes.slice_by_values("org", "H2")
        layers_v_z = boreholes.slice_by_values("lith", ["V", "Z"])

        assert len(layers_k.data) == 188
        assert len(layers_k.header) == 13
        assert len(layers_ks2.data) == 19
        assert len(layers_ks2.header) == 6
        assert len(layers_h2.data) == 3
        assert len(layers_h2.header) == 3
        assert len(layers_v_z.data) == 23
        assert len(layers_v_z.header) == 11

    @pytest.mark.unittest
    def test_inverted_slice_by_values(self, boreholes):
        layers_non_k = boreholes.slice_by_values("lith", "K", invert=True)
        layers_non_ks2 = boreholes.slice_by_values("lith_comb", "Ks2", invert=True)
        layers_non_h2 = boreholes.slice_by_values("org", "H2", invert=True)
        layers_non_v_z = boreholes.slice_by_values("lith", ["V", "Z"], invert=True)

        assert len(layers_non_k.data) == 162
        assert len(layers_non_k.header) == 13
        assert len(layers_non_ks2.data) == 331
        assert len(layers_non_ks2.header) == 13
        assert len(layers_non_h2.data) == 347
        assert len(layers_non_h2.header) == 13
        assert len(layers_non_v_z.data) == 327
        assert len(layers_non_v_z.header) == 13

    @pytest.mark.unittest
    def test_add_header_column_to_data(self, boreholes):
        boreholes.header["test_data"] = [i for i in range(len(boreholes.header))]
        boreholes.add_header_column_to_data("test_data")

        assert_allclose(boreholes.get("HB-6").data["test_data"], 0)
        assert_allclose(boreholes.get("HB-03").data["test_data"], 12)

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
    # def test_surface_level_update(self, boreholes, update_raster):
    #     boreholes.update_surface_level_from_raster(update_raster, how="replace")
    #     print("stop")
