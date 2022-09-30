from pysst import read_sst_cores, read_gef_cpt
from pysst import excel_to_parquet, csv_to_parquet
from pysst.borehole import BoreholeCollection
import geopandas as gpd

from time import perf_counter

# excel_to_parquet(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Eva.xlsx"
# )

# collection = read_sst_cores(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Eva.parquet"
# )
# collection.to_vtk("", vertical_factor=0.01)

# collection2 = read_sst_cores(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes.parquet",
# )

# collection.append(collection2)

# cpts = read_gef_cpt(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Data\van HHNK\sonderingen"
# )

deklaag_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen_wellen.parquet"
)
points = gpd.read_file(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\wellen\wellen_nl_RD.gpkg"
)
gmm = gpd.read_file(
    r"c:\Users\onselen\Lokale data\bro-geomorfologischekaart.gpkg",
    layer="view_geomorphological_area",
)


for i, point in points.iterrows():
    cores = deklaag_cores.select_from_points(point, buffer=800)
    point_simple = gpd.GeoDataFrame(geometry=[point.geometry])
    landform = gpd.sjoin(point_simple, gmm)["landformsubgroup_code"]
    cores_landforms = gpd.sjoin(cores.header, gmm)["landformsubgroup_code"]
    if not len(landform) == 0:
        cores_indices = cores_landforms[cores_landforms.values == landform.values].index
        ok_boreholes = BoreholeCollection(cores.data.loc[cores_indices])

print("stop")

# tic = perf_counter()
# dino = read_sst_cores(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
# )
# dino_selected = dino.select_from_polygons(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\dijk_shapes\Dijken_buffer_dissolved.gpkg"
# )
# dino_selected.to_shape(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen.shp",
#     selected=True,
# )
# toc = perf_counter()
# print(toc - tic)
# tic
