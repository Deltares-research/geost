from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

#############################################

all_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)

selected_cores = all_cores.select_from_bbox(12000, 18000, 382000, 386500)
selected_cores.to_vtm(
    r"p:\11203725-modelberekeningen\04_multibeam\01_data\boreholes.vtm",
    ["lith", "zm", "ak", "as", "az", "shells", "strat_2003"],
    radius=2,
)
