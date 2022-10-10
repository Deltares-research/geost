from pysst import read_sst_cores, read_gef_cpt
from pysst import excel_to_parquet, csv_to_parquet
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

from time import perf_counter

gefs = read_gef_cpt(
    r"n:\Projects\11207000\11207357\B. Measurements and calculations\Grondonderzoek Wiertsema\rapportage wiertsema_81175-1-r84010-geotechnisch-onderzoek-pdf_2022-06-14_1151\81175_Veldwerk_data\Sonderingen\GEF\Alleen sonderingen"
)

gefs.header


# excel_to_parquet(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Eva.xlsx"
# )

# collection = read_sst_cores(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Eva.parquet"
# )
# collection.to_vtk("", vertical_factor=0.01)


# tic = perf_counter()
# dino = read_sst_cores(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
# )
# toc = perf_counter()
# print(toc - tic)
# tic
