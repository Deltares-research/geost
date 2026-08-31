ModelDataArray
==================
.. currentmodule:: geost.models.model_array

This is the GeoST extension for `xarray.DataArray`. Available after importing `geost`. For
example:

.. code-block:: python

   import numpy as np
   import xarray as xr

   import geost

   # Create example model data array with 3D data
   da = xr.DataArray(
       np.random.rand(3, 3, 4),
       dims=("y", "x", "z"),
       coords={"y": [2, 1, 0], "x": [0, 1, 2], "z": [0, 1, 2, 3]},
   )
   da.gst
   # Output:
   # <geost.models.model_array.ModelDataArray at 0x7f8c4e4c0d30>


Analysis
----------
.. autosummary::
   :toctree: generated/

   ModelDataArray.get_thickness
   ModelDataArray.most_common
   ModelDataArray.value_counts


Coordinate Reference System
-----------------------------
.. autosummary::
   :toctree: generated/

   ModelDataArray.write_crs


General
----------
.. autosummary::
   :toctree: generated/

   ModelDataArray.bounds
   ModelDataArray.resolution
   ModelDataArray.vertical_bounds


Selection
----------
.. autosummary::
   :toctree: generated/

   ModelDataArray.slice_depth_interval
   ModelDataArray.slice_xy


Spatial
------------------
.. autosummary::
   :toctree: generated/

   ModelDataArray.mask_geometries
   ModelDataArray.select_points
   ModelDataArray.select_along_lines
   ModelDataArray.select_within_bbox


Attributes
----------
.. autosummary::
   :toctree: generated/

   ModelDataArray.bottom
   ModelDataArray.crs
   ModelDataArray.model_type
   ModelDataArray.ndims
   ModelDataArray.shape
   ModelDataArray.surface_level
   ModelDataArray.top
   ModelDataArray.x
   ModelDataArray.x_dim
   ModelDataArray.y
   ModelDataArray.y_dim
   ModelDataArray.z
   ModelDataArray.z_dim
