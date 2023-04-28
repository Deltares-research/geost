from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

#############################################

all_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)
core_area = gpd.read_file(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-140 - Feedback SOS Do9 10 Sweco\data\segments.shp"
)

selected_cores = all_cores.select_with_lines(core_area, buffer=600)
selected_cores_with_peat = selected_cores.select_by_values("lith", "V")
selected_cores_with_peat.get_cumulative_layer_thickness(
    "lith", "V", include_in_header=True
)
selected_cores.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-140 - Feedback SOS Do9 10 Sweco\data\boreholes.shp"
)
selected_cores_with_peat.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-140 - Feedback SOS Do9 10 Sweco\data\boreholes_veen.shp"
)
