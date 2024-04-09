import geost

cores = geost.read_sst_cores(
    r"n:\Projects\11207500\11207907\B. Measurements and calculations\Analyse_L17H\borehole_data\boreholes_5km_50cm.parquet"
)
cores.change_horizontal_reference(32631, only_geometries=False)
raster = (
    r"p:\tgg-mariene-data\__UPDATES\SVN_CHECKOUTS\LAT_MSL\lat_400m_clip\w001001.adf"
)

cores.update_surface_level_from_raster(raster, "-")
cores.to_parquet(
    r"n:\Projects\11207500\11207907\B. Measurements and calculations\Analyse_L17H\borehole_data\boreholes_5km_50cm_LAT.parquet"
)
cores.to_geoparquet(
    r"n:\Projects\11207500\11207907\B. Measurements and calculations\Analyse_L17H\borehole_data\boreholes_5km_50cm_LAT.geoparquet"
)
