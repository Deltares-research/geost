from geost.export import vtk


class VoxelModel:
    def __init__(self, ds):
        self.ds = ds

    def to_pyvista_grid(
        self, data_vars: str | list[str] = None, structured: bool = True
    ):  # NOTE: Method will differ between voxel and layer model
        """
        Convert the VoxelModel to a PyVista grid.

        Parameters
        ----------
        data_vars : str | list[str], optional
            String representing one data variable or list of data variables to include
            in the PyVista grid. If None, all data variables are included. The default
            is None.
        structured : bool, optional
            If True, convert to a structured grid. If False, convert to an unstructured
            grid. The default is True.

        Returns
        -------
        pyvista.UnstructuredGrid or pyvista.StructuredGrid
            PyVista grid representation of the VoxelModel.

        """
        if data_vars is None:
            data_vars = self.ds.data_vars
        elif isinstance(data_vars, str):
            data_vars = [data_vars]

        if structured:
            return vtk.voxelmodel_to_pyvista_structured(
                self.ds,
                self.resolution,
                displayed_variables=data_vars,
            )
        else:
            return vtk.voxelmodel_to_pyvista_unstructured(
                self.ds,
                self.resolution,
                displayed_variables=data_vars,
            )
