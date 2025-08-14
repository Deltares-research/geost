Input/Output
============

.. currentmodule:: geost

BRO api
-------
.. autosummary::
   :toctree: generated/

   bro_api_read


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

   read_bhrgt
   read_bhrg
   read_bhrp
   read_cpt
   read_gef_cpts
   read_sfr


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
