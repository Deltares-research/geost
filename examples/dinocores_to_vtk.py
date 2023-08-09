from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

#############################################

all_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)

selected_cores = all_cores.select_by_depth(end_max=-40).select_by_values(
    "strat_2003", ["PE", "PENI"]
)
selected_cores.get_cumulative_layer_thickness()
selected_cores.to_vtm(
    r"p:\11203725-modelberekeningen\04_multibeam\01_data\boreholes.vtm",
    ["lith", "zm", "ak", "as", "az", "shells", "strat_2003"],
    radius=2,
)
