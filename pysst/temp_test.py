from pysst import read_sst_cores, read_gef_cpt
from pysst import excel_to_parquet, csv_to_parquet

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

tic = perf_counter()
dino = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)
dino_selected = dino.select_from_polygon(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\dijk_shapes\Waal_buffer_l.gpkg"
)
dino_selected.to_shape(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen.shp",
    selected=True,
)
toc = perf_counter()
print(toc - tic)
tic
