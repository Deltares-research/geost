BRO
===

.. currentmodule:: geost.bro

BRO api
-------
.. autosummary::
   :toctree: generated/

   BroApi
   BroApi.get_objects
   BroApi.search_objects_in_bbox
   BroApi.apis
   BroApi.document_types


BRO GeoTOP
----------
Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   GeoTop
   GeoTop.from_netcdf
   GeoTop.from_opendap


Methods
~~~~~~~
.. autosummary::
   :toctree: generated/

   GeoTop.select
   GeoTop.select_bottom
   GeoTop.select_by_values
   GeoTop.select_index
   GeoTop.select_surface_level
   GeoTop.select_top
   GeoTop.select_with_line
   GeoTop.select_with_points
   GeoTop.select_within_bbox
   GeoTop.select_within_polygons
   GeoTop.slice_depth_interval
   GeoTop.zslice_to_tiff


Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   GeoTop.crs
   GeoTop.horizontal_bounds
   GeoTop.resolution
   GeoTop.shape
   GeoTop.sizes
   GeoTop.variables
   GeoTop.vertical_bounds
   GeoTop.xmax
   GeoTop.xmin
   GeoTop.ymax
   GeoTop.ymin
   GeoTop.zmax
   GeoTop.zmin
