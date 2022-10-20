from pysst import read_sst_cores, read_gef_cpt
from pysst import excel_to_parquet, csv_to_parquet
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

from time import perf_counter

gefs = read_gef_cpt(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Data\van HHNK\sonderingen"
)

gefs.add_lithology()
gefs.to_parquet(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Data\van HHNK\cpts.parquet"
)
gefs.header


# excel_to_parquet(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Eva.xlsx"
# )

# collection = read_sst_cores(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Eva.parquet"
# )
# collection.to_vtk("", vertical_factor=0.01)


tic = perf_counter()
dino = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)
toc = perf_counter()
print(toc - tic)
dino_selected = dino.select_from_lines(
    gpd.read_file(
        r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-88 - Review SOS RHDHV_deelopdracht_3b\data\segmenten_85.shp"
    ),
    buffer=600,
)
dino_selected.to_geoparquet(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-88 - Review SOS RHDHV_deelopdracht_3b\data\cores_rhdhv.parquet"
)
tic
