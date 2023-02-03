from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

#############################################

all_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)
core_area = gpd.read_file(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-130 - Feedback Arcadis DO9 DO4 extra werk\data\segmenten.shp"
)

selected_cores = all_cores.select_with_lines(core_area, buffer=600)
selected_cores_with_peat = selected_cores.select_by_values({"lith": ["V"]})
selected_cores.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-130 - Feedback Arcadis DO9 DO4 extra werk\data\boreholes.shp"
)
selected_cores_with_peat.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-130 - Feedback Arcadis DO9 DO4 extra werk\data\boreholes_veen.shp"
)
