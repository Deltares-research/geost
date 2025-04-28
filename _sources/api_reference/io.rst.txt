Input/Output
============

.. currentmodule:: geost

BRO api
-------
.. autosummary::
   :toctree: generated/

   get_bro_objects_from_bbox
   get_bro_objects_from_geometry


Data in tabular format (e.g. csv, parquet)
------------------------------------------
.. autosummary::
   :toctree: generated/

   read_borehole_table
   read_cpt_table
   read_nlog_cores
   read_uullg_tables
   base.Collection.to_csv
   base.Collection.to_parquet


Data from specific file formats
-------------------------------
.. autosummary::
   :toctree: generated/

   read_gef_cpts


GIS vector files
----------------
.. autosummary::
   :toctree: generated/

   read_collection_geopackage
   base.Collection.to_geopackage
   base.Collection.to_geoparquet
   base.Collection.to_shape
   io.Geopackage

Data in binary format (e.g. pickle)
------------------------------------
.. autosummary::
   :toctree: generated/

   read_pickle
