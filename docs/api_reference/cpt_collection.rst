CptCollection
==============
.. currentmodule:: geost.base

Constructor
-----------
.. autosummary::
   :toctree: generated/

   CptCollection


Analysis
----------
.. autosummary::
   :toctree: generated/

   CptCollection.get_area_labels
   CptCollection.get_cumulative_thickness
   CptCollection.get_layer_top


Coordinate Reference System
-----------------------------
.. autosummary::
   :toctree: generated/

   CptCollection.change_horizontal_reference
   CptCollection.change_vertical_reference


Export
-----------------------------
.. autosummary::
   :toctree: generated/

   CptCollection.to_csv
   CptCollection.to_datafusiontools
   CptCollection.to_geopackage
   CptCollection.to_geoparquet
   CptCollection.to_pyvista_cylinders
   CptCollection.to_pyvista_grid
   CptCollection.to_parquet
   CptCollection.to_shape


Generic
-----------------------------
.. autosummary::
   :toctree: generated/

   CptCollection.add_header_column_to_data
   CptCollection.check_header_to_data_alignment
   CptCollection.reset_header


Selection
----------
.. autosummary::
   :toctree: generated/

   CptCollection.get
   CptCollection.select_by_condition
   CptCollection.select_by_depth
   CptCollection.select_by_length
   CptCollection.select_by_values
   CptCollection.slice_by_values
   CptCollection.slice_depth_interval


Spatial Selection
------------------
.. autosummary::
   :toctree: generated/

   CptCollection.select_with_lines
   CptCollection.select_with_points
   CptCollection.select_within_bbox
   CptCollection.select_within_polygons


Attributes
----------
.. autosummary::
   :toctree: generated/

   CptCollection.data
   CptCollection.has_inclined
   CptCollection.header
   CptCollection.horizontal_reference
   CptCollection.n_points
   CptCollection.vertical_reference
