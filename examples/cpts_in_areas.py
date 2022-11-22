from pysst import read_gef_cpts
import geopandas as gpd

gef_collection = read_gef_cpts(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Purmerend DualEM\Model training\Training_CPTs"
)
gef_collection.add_ic()
gef_collection.add_lithology()

gef_collection.to_shape(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Purmerend DualEM\Model training\Training locations\CPT_locations.shp"
)
gef_collection.to_parquet(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Purmerend DualEM\Model training\Training_CPTs\CPT_data.parquet"
)


# Import polygon file
gdf = gpd.read_file(r"c:\path\to\shapefileorgeopackage")
# Select only GEFs in area if required
gdf_in_area = gef_collection.select_from_polygons(gdf)
# Find which labels of the polygons correspond to the CPT's
labels = gef_collection.get_area_labels(gdf_in_area, "column_name_to_get_labels_from")

# Write locations to multipoint shapefile/geopackage
gef_collection.to_shape(r"c:\newfileloc.gpkg")
# Or try geoparquet which is much smaller and faster. Requires GDAL 3.5+ and Qgis 3.26+ for viewing.
gef_collection.to_geoparquet(r"c:\newfileloc.parquet")
