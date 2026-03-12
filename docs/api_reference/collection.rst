Collection
==================
.. currentmodule:: geost.base

Constructor
-----------
.. autosummary::
   :toctree: generated/

   Collection


Analysis
----------
.. autosummary::
   :toctree: generated/

   Collection.get_cumulative_thickness
   Collection.get_layer_base
   Collection.get_layer_top


Coordinate Reference System
-----------------------------
.. autosummary::
   :toctree: generated/

   Collection.change_horizontal_reference
   Collection.change_vertical_reference


Export
-----------------------------
.. autosummary::
   :toctree: generated/

   Collection.to_csv
   Collection.to_datafusiontools
   Collection.to_geopackage
   Collection.to_geoparquet
   Collection.to_kingdom
   Collection.to_pyvista_cylinders
   Collection.to_pyvista_grid
   Collection.to_parquet
   Collection.to_qgis3d
   Collection.to_shape


Generic
-----------------------------
.. autosummary::
   :toctree: generated/

   Collection.add_header_column_to_data
   Collection.check_header_to_data_alignment
   Collection.reset_header


Selection
----------
.. autosummary::
   :toctree: generated/

   Collection.get
   Collection.select_by_condition
   Collection.select_by_depth
   Collection.select_by_length
   Collection.select_by_values
   Collection.slice_by_values
   Collection.slice_depth_interval


Spatial
------------------
.. autosummary::
   :toctree: generated/

   Collection.select_with_lines
   Collection.select_with_points
   Collection.select_within_bbox
   Collection.select_within_polygons
   Collection.spatial_join


Attributes
----------
.. autosummary::
   :toctree: generated/

   Collection.data
   Collection.has_inclined
   Collection.header
   Collection.horizontal_reference
   Collection.n_points
   Collection.vertical_reference
