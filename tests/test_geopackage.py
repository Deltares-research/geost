import sqlite3

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from geost.io import Geopackage


class TestGeopackage:
    @pytest.mark.unittest
    def test_get_connection(self, simple_soilmap_gpkg):
        gp = Geopackage(simple_soilmap_gpkg)
        gp.get_connection()
        assert isinstance(gp.connection, sqlite3.Connection)

    @pytest.mark.unittest
    def test_layers(self, simple_soilmap_gpkg):
        gp = Geopackage(simple_soilmap_gpkg)
        layers = gp.layers()
        assert isinstance(layers, pd.DataFrame)
        assert_array_equal(layers["name"], ["soilarea", "soilarea_soilunit"])
        assert_array_equal(layers["geometry_type"], ["Polygon", None])

    @pytest.mark.unittest
    def test_context_manager(self, simple_soilmap_gpkg):
        gp = Geopackage(simple_soilmap_gpkg)
        assert gp.connection is None
        with gp:
            assert isinstance(gp.connection, sqlite3.Connection)
        assert gp.connection is None

    @pytest.mark.unittest
    def test_get_cursor(self, simple_soilmap_gpkg):
        with Geopackage(simple_soilmap_gpkg) as gp:
            cursor = gp._get_cursor()
            assert isinstance(cursor, sqlite3.Cursor)

    @pytest.mark.unittest
    def test_get_column_names(self, simple_soilmap_gpkg):
        with Geopackage(simple_soilmap_gpkg) as gp:
            columns = gp.get_column_names("soilarea_soilunit")
            assert_array_equal(columns, ["fid", "maparea_id", "soilunit_code"])

    @pytest.mark.unittest
    def test_read_table(self, simple_soilmap_gpkg):
        test_table = "soilarea_soilunit"
        with Geopackage(simple_soilmap_gpkg) as gp:
            table = gp.read_table(test_table)
            assert isinstance(table, pd.DataFrame)
            assert_array_equal(table.columns, ["fid", "maparea_id", "soilunit_code"])

            table = gp.table_head(test_table)
            assert len(table) == 5

    @pytest.mark.unittest
    def test_query(self, simple_soilmap_gpkg):
        with Geopackage(simple_soilmap_gpkg) as gp:
            query = "SELECT * FROM soilarea_soilunit"
            table = gp.query(query)
        assert isinstance(table, pd.DataFrame)
        assert_array_equal(table.columns, ["fid", "maparea_id", "soilunit_code"])

        with Geopackage(simple_soilmap_gpkg) as gp:
            query = "SELECT * FROM soilarea_soilunit WHERE fid = 1"
            table = gp.query(query, outcolumns=["A", "B", "C"])
        assert_array_equal(table.columns, ["A", "B", "C"])
