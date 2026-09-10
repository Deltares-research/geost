import geost
import geopandas as gpd

from shapely.geometry import Point

# model = geost.read_model_netcdf(
#     r"C:\data\freshem\temp\resistivity_model_with_lithoclasses_zuidA1_v20260909.nc"
# )

# grid = model.gst.to_pyvista_grid()
# grid.save(r"C:\data\freshem\temp\resistivity_model_with_lithoclasses_zuidA1_v20260909.vtk")


table = geost.read_table(r"c:\data\combined_boreholes_uu_dino_bhrgt_20260807.parquet")

geometries = [
    Point(xy) for xy in zip([100000, 120000, 140000], [400000, 420000, 440000])
]
gdf = gpd.GeoDataFrame(geometry=geometries)

pairs = table.find_point_pairs(gdf, max_distance=100, n_points=2, return_distance=True)
