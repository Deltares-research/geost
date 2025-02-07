from enum import StrEnum
from pathlib import Path

import geopandas as gpd

from geost.io import Geopackage


class CptTables(StrEnum):
    geotechnical_cpt_survey = "geotechnical_cpt_survey"
    delivered_location = "delivered_location"
    bro_location = "bro_location"
    bro_point = "bro_point"
    delivered_vertical_position = "delivered_vertical_position"
    cone_penetrometer_survey = "cone_penetrometer_survey"
    trajectory = "trajectory"
    processing = "processing"
    cone_penetrometer = "cone_penetrometer"
    zero_load_measurement = "zero_load_measurement"
    parameters = "parameters"
    cone_penetration_test = "cone_penetration_test"
    cone_penetration_test_result = "cone_penetration_test_result"
    dissipation_test = "dissipation_test"
    dissipation_test_result = "dissipation_test_result"
    additional_investigation = "additional_investigation"
    removed_layer = "removed_layer"
    registration_history = "registration_history"
    nga_properties = "nga_properties"


class BroCptGeopackage:
    def __init__(self, gdf: gpd.GeoDataFrame, db: Geopackage):
        self.gdf = gdf
        self.db = db

    @classmethod
    def from_geopackage(cls, file: str | Path, **gpd_kwargs):
        if "fid_as_index" not in gpd_kwargs:  # Needs to retain index for db selections
            gpd_kwargs["fid_as_index"] = True

        if "layer" in gpd_kwargs:
            raise ValueError("Layer cannot be passed as a Geopandas keyword argument.")

        gdf = gpd.read_file(file, layer=CptTables.geotechnical_cpt_survey, **gpd_kwargs)
        db = Geopackage(file)
        return cls(gdf, db)
