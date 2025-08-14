from pathlib import Path

import geopandas as gpd
import pytest
from numpy.testing import assert_array_equal

from geost.bro import BroCptGeopackage
from geost.io import Geopackage


class TestBroCptGeopackage:
    @pytest.mark.unittest
    def test_from_geopackage(self, bro_cpt_gpkg: Path):
        bro_cpt = BroCptGeopackage.from_geopackage(bro_cpt_gpkg)

        assert isinstance(bro_cpt.gdf, gpd.GeoDataFrame)
        assert isinstance(bro_cpt.db, Geopackage)
        assert bro_cpt.gdf.index.name == "fid"
        assert_array_equal(
            bro_cpt.gdf.columns,
            [
                "bro_id",
                "quality_regime",
                "delivery_accountable_party",
                "delivery_context",
                "survey_purpose",
                "research_report_date",
                "cpt_standard",
                "additional_investigation_performed",
                "applied_transformation",
                "geometry",
            ],
        )

        with pytest.raises(ValueError):
            BroCptGeopackage.from_geopackage(
                bro_cpt_gpkg, layer="geotechnical_cpt_survey"
            )

        bro_cpt = BroCptGeopackage.from_geopackage(bro_cpt_gpkg, fid_as_index=False)
        assert bro_cpt.gdf.index.name is None

    @pytest.mark.unittest
    def test_select_location_info(self, bro_cpt_gpkg: Path):
        bro_cpt = BroCptGeopackage.from_geopackage(bro_cpt_gpkg)
        with pytest.raises(NotImplementedError):
            bro_cpt.select_location_info()
