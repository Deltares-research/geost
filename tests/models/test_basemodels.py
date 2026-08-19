import re

import pytest
import pyvista as pv


@pytest.mark.xfail(reason="`VoxelModel` is deprecated")
class TestVoxelModel:
    """
    This test class can be deprecated. Test functions below are not yet transferred to
    `ModelDataset` and `ModelDataArray` as the `to_pyvista_grid` method has not yet been
    implemented. When done, this can all be removed.

    """

    @pytest.mark.unittest
    def test_to_pyvista_structured(self, to_remove_voxelmodel):
        vms_single_var = to_remove_voxelmodel.to_pyvista_grid(data_vars=["strat"])
        assert isinstance(vms_single_var, pv.ImageData)

        vms_multi_var = to_remove_voxelmodel.to_pyvista_grid()
        assert isinstance(vms_multi_var, pv.ImageData)

    @pytest.mark.unittest
    def test_to_pyvista_unstructured(self, to_remove_voxelmodel):
        vmu_single_var = to_remove_voxelmodel.to_pyvista_grid(
            data_vars=["strat"], structured=False
        )
        assert isinstance(vmu_single_var, pv.UnstructuredGrid)

        vmu_multi_var = to_remove_voxelmodel.to_pyvista_grid(structured=False)
        assert isinstance(vmu_multi_var, pv.UnstructuredGrid)

    @pytest.mark.unittest
    def test_to_pyvista_unstructured_problematic_dims(self, to_remove_voxelmodel):
        # Wrong order of dimensions leads to automatic transposing, not an error!

        # Why are the five line below in this test? The same happens in the test above.
        vmu_wrong_order = to_remove_voxelmodel.to_pyvista_grid(structured=False)
        assert isinstance(vmu_wrong_order, pv.UnstructuredGrid)

        # Missing z-dimension leads to an error and no file is created.
        to_remove_voxelmodel.ds = to_remove_voxelmodel.ds.drop_vars("z")
        with pytest.raises(Exception) as error_info:
            to_remove_voxelmodel.to_pyvista_grid()
        assert error_info.errisinstance(ValueError)
        assert error_info.match(
            re.escape(
                "Dataset must contain 'z' dimension. Make sure that this "
                "spatial dimension exists in the dataset or if it has a different "
                "name use xarray.Dataset.rename() to rename the corresponding "
                "dimension to 'z'."
            )
        )
