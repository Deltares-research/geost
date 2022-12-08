from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

#############################################

all_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)
core_area = gpd.read_file(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-127 - Feedback SOS Sweco DO8 extra werk\data\segmenten.shp"
)

selected_cores = all_cores.select_from_lines(core_area, buffer=600)
selected_cores.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-127 - Feedback SOS Sweco DO8 extra werk\data\boreholes.shp"
)
