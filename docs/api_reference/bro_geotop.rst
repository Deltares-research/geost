BRO GeoTOP
==========

.. currentmodule:: geost.bro.bro_geotop

GeoTop
------

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


StratGeotop
-----------
.. autosummary::
   :toctree: generated/

   StratGeotop


Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   StratGeotop.holocene
   StratGeotop.channel
   StratGeotop.older
   StratGeotop.anthropogenic


Methods
~~~~~~~
.. autosummary::
   :toctree: generated/

   StratGeotop.select_units
   StratGeotop.select_values


UnitEnums
---------
`UnitEnum` objects for stratigraphic units and corresponding numbers in GeoTOP for four main groups.

.. autosummary::
   :toctree: generated/

   HoloceneUnits
   ChannelBeltUnits
   OlderUnits
   AntropogenicUnits
