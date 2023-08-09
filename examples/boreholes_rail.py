import geopandas as gpd
import numpy as np

#############################################
import pandas as pd

from pysst import read_sst_cores
from pysst.base import PointDataCollection
from pysst.borehole import BoreholeCollection

all_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)
core_area = gpd.read_file(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-145 - Feedback Arcadis Maaslijn\data\segments.shp"
)

selected_cores = all_cores.select_with_lines(core_area, buffer=600)
selected_cores_with_peat = selected_cores.select_by_values("lith", "V")
selected_cores_with_peat.get_cumulative_layer_thickness(
    "lith", "V", include_in_header=True
)
cores_first_5m = selected_cores.slice_depth_interval(
    lower_boundary=5, upper_boundary=0, vertical_reference="depth"
)
cores_first_5m = cores_first_5m.select_by_values("lith", ["L", "K"])
cores_first_5m.get_cumulative_layer_thickness(
    "lith", ["L", "K"], include_in_header=True
)
cores_first_5m.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-145 - Feedback Arcadis Maaslijn\data\boreholes_KL_5m.shp"
)

selected_cores.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-145 - Feedback Arcadis Maaslijn\data\boreholes.shp"
)
selected_cores_with_peat.to_shape(
    r"n:\Projects\11207500\11207576\B. Measurements and calculations\PRORAIL-145 - Feedback Arcadis Maaslijn\data\boreholes_veen.shp"
)
