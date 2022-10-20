import pandas as pd
from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

# Producing features and a target for use in a Machine Learning model (Deltares Datafusiontools)

deklaag_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen_binnendijk.parquet"
)

morfologie = gpd.read_file(
    r"c:\Users\onselen\Lokale data\bro-geomorfologischekaart.gpkg",
    layer="view_geomorphological_area",
)

bodem = gpd.read_file(
    r"c:\Users\onselen\Lokale data\Bodemkaart_2014.gpkg",
)

geologie = gpd.read_file(
    r"c:\Users\onselen\Lokale data\GKNederlandGeolVlak.gpkg",
)


# Features extracted from maps
areas_morfologie = deklaag_cores.get_area_labels(morfologie, "landformsubgroup_code")
areas_geologie = deklaag_cores.get_area_labels(geologie, "CODE")
areas_bodem = deklaag_cores.get_area_labels(bodem, "BODEM1")
# Features from collection of boreholes
x = deklaag_cores.header["x"]
y = deklaag_cores.header["y"]
maaiveld = deklaag_cores.header["mv"]
# Combine features
features = pd.concat(
    [areas_morfologie, areas_geologie, areas_bodem, x, y, maaiveld], axis=1
)

# Target
cover_layer = deklaag_cores.cover_layer_thickness()
mask = np.isfinite(cover_layer["cover_thickness"])
cover_layer = cover_layer[mask]
features = features[mask]

features.to_csv(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\Experiment datafusion\features.csv",
    index=False,
    header=True,
)
cover_layer.to_csv(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\Experiment datafusion\target.csv",
    index=False,
    header=True,
)
