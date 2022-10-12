from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

deklaag_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen_binnendijk.parquet"
)
cover_layer = deklaag_cores.cover_layer_thickness(allow_partial_cover_layers=True)

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


areas_morfologie = deklaag_cores.add_area_labels(morfologie, "landformsubgroup_code")
areas_geologie = deklaag_cores.add_area_labels(geologie, "CODE")
areas_bodem = deklaag_cores.add_area_labels(bodem, "BODEM1")

test = 1
