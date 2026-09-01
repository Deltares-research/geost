ModelDataset
==================
.. currentmodule:: geost.models.model_dataset

This is the GeoST extension for `xarray.Dataset`. Available after importing `geost`. For
example:

.. code-block:: python

   import numpy as np
   import xarray as xr

   import geost

   # Create example model dataset with 3D data variable
   ds = xr.Dataset(
       data_vars={"value": (("y", "x", "z"), np.random.rand(3, 3, 4))},
       coords={"y": [2, 1, 0], "x": [0, 1, 2], "z": [0, 1, 2, 3]},
   )
   ds.gst
   # Output:
   # <geost.models.model_dataset.ModelDataset at 0x7f8c4e4c0d30>


Analysis
----------
.. autosummary::
   :toctree: generated/

   ModelDataset.get_thickness
   ModelDataset.most_common


Coordinate Reference System
-----------------------------
.. autosummary::
   :toctree: generated/

   ModelDataset.write_crs


General
----------
.. autosummary::
   :toctree: generated/

   ModelDataset.bounds
   ModelDataset.resolution
   ModelDataset.vertical_bounds


Selection
----------
.. autosummary::
   :toctree: generated/

   ModelDataset.slice_depth_interval
   ModelDataset.slice_xy


Spatial
------------------
.. autosummary::
   :toctree: generated/

   ModelDataset.mask_geometries
   ModelDataset.select_points
   ModelDataset.select_along_lines
   ModelDataset.select_within_bbox


Attributes
----------
.. autosummary::
   :toctree: generated/

   ModelDataset.bottom
   ModelDataset.crs
   ModelDataset.model_type
   ModelDataset.ndims
   ModelDataset.shape
   ModelDataset.surface_level
   ModelDataset.top
   ModelDataset.x
   ModelDataset.x_dim
   ModelDataset.y
   ModelDataset.y_dim
   ModelDataset.z
   ModelDataset.z_dim
