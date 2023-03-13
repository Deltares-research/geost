import pytest
import pickle
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd

from pysst.borehole import BoreholeCollection
from pysst.export import vtk


class TestExport:
    export_folder = Path(__file__).parent / "data"

    @pytest.fixture
    def borehole_collection(self):
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

        return BoreholeCollection(dataframe)

    @pytest.mark.unittest
    def test_to_parquet(self, borehole_collection):
        out_file = self.export_folder.joinpath("test_output_file.parquet")
        borehole_collection.to_parquet(out_file)
        assert out_file.is_file()
        out_file.unlink()

    @pytest.mark.unittest
    def test_to_csv(self, borehole_collection):
        out_file = self.export_folder.joinpath("test_output_file.csv")
        borehole_collection.to_csv(out_file)
        assert out_file.is_file()
        out_file.unlink()

    @pytest.mark.unittest
    def test_to_shape(self, borehole_collection):
        out_file = self.export_folder.joinpath("test_output_file.shp")
        cpg_file = self.export_folder.joinpath("test_output_file.cpg")
        dbf_file = self.export_folder.joinpath("test_output_file.dbf")
        shx_file = self.export_folder.joinpath("test_output_file.shx")
        borehole_collection.to_shape(out_file)
        assert out_file.is_file()
        assert cpg_file.is_file()
        assert dbf_file.is_file()
        assert shx_file.is_file()
        out_file.unlink()
        cpg_file.unlink()
        dbf_file.unlink()
        shx_file.unlink()

    @pytest.mark.unittest
    def test_to_geoparquet(self, borehole_collection):
        out_file = self.export_folder.joinpath("test_output_file.geoparquet")
        borehole_collection.to_geoparquet(out_file)
        assert out_file.is_file()
        out_file.unlink()

    @pytest.mark.unittest
    def test_to_ipf(self, borehole_collection):
        # TODO
        pass

    @pytest.mark.unittest
    def test_vtk_prepare_borehole(self, borehole_collection):
        prepared_borehole = vtk.prepare_borehole(borehole_collection.data, 1.0)
        target = np.array(
            [
                [139370, 455540, 1.0],
                [139370, 455540, 0.5],
                [139370, 455540, 0.0],
                [139370, 455540, -0.5],
                [139370, 455540, -1.5],
                [139370, 455540, -2.0],
                [139370, 455540, -2.5],
                [139370, 455540, -3.0],
                [139370, 455540, -3.2],
                [139370, 455540, -3.6],
                [139370, 455540, -4.0],
            ]
        )
        assert_array_equal(prepared_borehole, target)

    @pytest.mark.unittest
    def test_vtk_borehole_to_multiblock(self, borehole_collection):
        multiblock = vtk.borehole_to_multiblock(
            borehole_collection.data,
            ["data_string", "data_int", "data_float"],
            0.5,
            1.0,
        )
        assert multiblock.n_blocks == 1
        assert multiblock.bounds == (139369.5, 139370.5, 455539.5, 455540.5, -4.0, 1.0)
        assert multiblock[0].n_arrays == 4
        assert multiblock[0].n_cells == 22
        assert multiblock[0].n_points == 260

    @pytest.mark.unittest
    def test_to_vtm(self, borehole_collection):
        out_file = self.export_folder.joinpath("test_output_file.vtm")
        out_folder = self.export_folder.joinpath("test_output_file")
        borehole_collection.to_vtm(out_file, ["data_string", "data_int", "data_float"])
        assert out_file.is_file()
        assert out_folder.is_dir()
        assert (out_folder / "test_output_file_0.vtp").is_file()
        out_file.unlink()
        (out_folder / "test_output_file_0.vtp").unlink()
        out_folder.rmdir()

    @pytest.mark.unittest
    def test_to_datafusiontools_list(self, borehole_collection):
        dft_objects = borehole_collection.to_datafusiontools(
            ["data_string", "data_int", "data_float"]
        )
        assert len(dft_objects) == 1
        assert [var.label for var in dft_objects[0].variables] == [
            "data_int",
            "data_float",
            "data_string_K",
            "data_string_Ks2",
            "data_string_Ks3",
            "data_string_Kz",
            "data_string_V",
            "data_string_Z",
            "data_string_Zk",
            "data_string_Zs",
        ]
        assert_array_equal(
            dft_objects[0].variables[0].value,
            borehole_collection.data["data_int"].values,
        )
        assert_array_almost_equal(
            dft_objects[0].variables[1].value,
            borehole_collection.data["data_float"].values,
        )
        assert_array_equal(
            dft_objects[0].variables[2].value, np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        )
        assert dft_objects[0].location.x == 139370
        assert dft_objects[0].location.y == 455540
        assert dft_objects[0].location.z == 1.0

    @pytest.mark.unittest
    def test_to_datafusiontools_pickle(self, borehole_collection):
        out_file = self.export_folder.joinpath("test_output_file.pickle")
        borehole_collection.to_datafusiontools(
            ["data_string", "data_int", "data_float"], out_file=out_file
        )
        assert out_file.is_file()
        with open(out_file, "rb") as f:
            dft_objects = pickle.load(f)
        assert len(dft_objects) == 1
        assert [var.label for var in dft_objects[0].variables] == [
            "data_int",
            "data_float",
            "data_string_K",
            "data_string_Ks2",
            "data_string_Ks3",
            "data_string_Kz",
            "data_string_V",
            "data_string_Z",
            "data_string_Zk",
            "data_string_Zs",
        ]
        assert_array_equal(
            dft_objects[0].variables[0].value,
            borehole_collection.data["data_int"].values,
        )
        assert_array_almost_equal(
            dft_objects[0].variables[1].value,
            borehole_collection.data["data_float"].values,
        )
        assert_array_equal(
            dft_objects[0].variables[2].value, np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        )
        assert dft_objects[0].location.x == 139370
        assert dft_objects[0].location.y == 455540
        assert dft_objects[0].location.z == 1.0
        out_file.unlink()
