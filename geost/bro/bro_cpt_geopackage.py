from enum import StrEnum
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geost.io import Geopackage


class CptTables(StrEnum):  # pragma: no cover
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
    """
    Class to handle the Bro CPT Geopackage file for data selections and facilitate making
    combinations between the different data tables in the BRO CPT geopackage.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the spatial locations of the CPT data. The "find key" per
        "bro_id" to each related tables in the Geopackage as index (index name: "fid").
    db : :class:`~geost.io.Geopackage`
        Geost Geopackage instance to handle the database connections and queries.
    """

    def __init__(self, gdf: gpd.GeoDataFrame, db: Geopackage):
        self.gdf = gdf
        self.db = db

    @classmethod
    def from_geopackage(cls, file: str | Path, **gpd_kwargs):
        """
        Create an instance of `BroCptGeopackage` from a GeoPackage file.

        Parameters
        ----------
        file : str or Path
            The path to the GeoPackage file.
        **gpd_kwargs : dict
            Additional keyword arguments to pass to `gpd.read_file`. The 'fid_as_index'
            argument is set to True by default to retain the index fordatabase selections.
            The 'layer' argument should not be passed as it is set internally and this will
            raise a ValueError.

        Returns
        -------
        `BroCptGeopackage`

        Raises
        ------
        ValueError
            If the 'layer' argument is passed in `gpd_kwargs`. This argument is set internally
            to set the `gdf` attribute and cannot be overwritten.

        """
        if "fid_as_index" not in gpd_kwargs:  # Needs to retain index for db selections
            gpd_kwargs["fid_as_index"] = True

        if "layer" in gpd_kwargs:
            raise ValueError("Layer cannot be passed as a Geopandas keyword argument.")

        gdf = gpd.read_file(file, layer=CptTables.geotechnical_cpt_survey, **gpd_kwargs)
        db = Geopackage(file)
        return cls(gdf, db)

    def select_location_info(self) -> pd.DataFrame:  # pragma: no cover
        """
        Select the location information from the BRO CPT geopackage.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the location information.

        """
        raise NotImplementedError("This method is not implemented yet.")
        gcs = CptTables.geotechnical_cpt_survey
        dl = CptTables.delivered_location
        bl = CptTables.bro_location
        bp = CptTables.bro_point

        query = f"""
            SELECT
                GCS.fid,
                DL.horizontal_positioning_date, DL.horizontal_positioning_method,
                BL.crs,
                BP.x_or_lon, BP.y_or_lat
            FROM {gcs} GCS
            JOIN {dl} DL ON GCS.fid = DL.{gcs}_fk
            JOIN {bl} BL ON DL.{dl}_pk = BL.{dl}_fk
            JOIN {bp} BP ON BL.{bl}_pk = BP.{bl}_fk
            LIMIT 5
        """
        with self.db:
            self.db.query(query)
        pass
