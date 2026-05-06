GeostFrame ``.gst`` accessor
==============================

.. currentmodule:: geost.accessor

GeoST extends `Pandas <https://pandas.pydata.org/docs/development/extending.html>`__ and
`Geopandas <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html>`__
with the ``.gst`` accessor, which provides all GeoST-specific methods to generic Pandas ``DataFrame``
and Geopandas ``GeoDataFrame`` objects. The ``.gst`` accessor becomes available by importing
``geost`` like so:

.. code-block:: python

   import geost

Once imported, the ``.gst`` accessor can be used on any Pandas ``DataFrame`` or Geopandas ``GeoDataFrame``.
For example:

.. code-block:: python

    import pandas as pd

    # create a DataFrame with x and y coordinates for three points identified by 'nr'
    df = pd.DataFrame({"nr": ["a", "b", "c"], "y": [1.2, 2.3, 3.4], "x": [0.8, 1.9, 2.0]})
    df.gst.has_xy_columns
    # Output:
    # True


Constructor
-----------------------------

.. autosummary::
   :toctree: generated/

   GeostFrame

Analysis
----------
.. autosummary::
   :toctree: generated/

   GeostFrame.get_cumulative_thickness
   GeostFrame.get_layer_base
   GeostFrame.get_layer_top

Coordinate Reference System
-----------------------------
.. autosummary::
   :toctree: generated/

   GeostFrame.change_horizontal_reference
   GeostFrame.change_vertical_reference
   GeostFrame.transform_coordinates

Generic
-----------------------------
.. autosummary::
   :toctree: generated/

   GeostFrame.calculate_thickness
   GeostFrame.standardize_column_names
   GeostFrame.to_collection
   GeostFrame.to_header
   GeostFrame.validate

Export
-----------------------------
.. autosummary::
   :toctree: generated/

   GeostFrame.to_kingdom
   GeostFrame.to_pyvista_cylinders
   GeostFrame.to_pyvista_grid
   GeostFrame.to_qgis3d

Selection
-----------------------------
.. autosummary::
   :toctree: generated/

   GeostFrame.select_by_condition
   GeostFrame.select_by_elevation
   GeostFrame.select_by_length
   GeostFrame.select_by_values
   GeostFrame.slice_by_values
   GeostFrame.slice_depth_interval

Spatial
-----------------------------
.. autosummary::
   :toctree: generated/

   GeostFrame.select_with_lines
   GeostFrame.select_with_points
   GeostFrame.select_within_bbox
   GeostFrame.select_within_polygons
   GeostFrame.spatial_join


Attributes
----------
.. autosummary::
   :toctree: generated/

   GeostFrame.first_row_survey
   GeostFrame.has_depth_columns
   GeostFrame.has_geometry
   GeostFrame.has_xy_columns
   GeostFrame.last_row_survey
   GeostFrame.is_layered
