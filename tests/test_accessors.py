import geopandas as gpd
import pandas as pd
import pytest

from geost.accessors.accessor import DATA, HEADER


class TestHeaderAccessor:
    @pytest.fixture
    def point(self):
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy([0, 1], [0, 1]))

    @pytest.fixture
    def linestring(self):
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(
                ["LINESTRING (0 0, 1 1)", "LINESTRING (1 0, 0 1)"]
            )
        )

    @pytest.fixture
    def polygon(self):
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                    "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
                ]
            )
        )

    @pytest.fixture
    def invalid(self):
        return pd.DataFrame()  # No geometry column is invalid for the accessor.

    @pytest.mark.unittest
    def test_get_geom_type(self, invalid):
        with pytest.raises(
            TypeError, match="Header accessor only accepts GeoDataFrames."
        ):
            invalid.gsthd

    @pytest.mark.parametrize(
        "header",
        ["point", "linestring", "polygon"],
        ids=["point", "linestring", "polygon"],
    )
    def test_backend_selection(self, header, request):
        """
        Test to check whether the correct backend is selected based on the 'headertype'
        attribute.

        """
        if header == "polygon":
            with pytest.raises(
                TypeError, match="No Header backend available for polygon"
            ):
                request.getfixturevalue(header).gsthd
        else:
            backend = HEADER[header]
            header = request.getfixturevalue(header)
            assert isinstance(header.gsthd._backend, backend)


class TestDataAccessor:
    @pytest.mark.unittest
    def test_validate(self):
        """
        Test to check wheter an input DataFrame has the required 'datatype' attribute
        which is needed to choose the correct backend for the accessor.

        """
        df = pd.DataFrame()
        with pytest.raises(
            AttributeError,
            match="Data has no attribute 'datatype', geodata accessor cannot choose backend",
        ):
            df.gstda  # Triggers `self._validate(gdf)` in `Data.__init__`

    @pytest.mark.parametrize(
        "datatype",
        ["layered", "discrete", "invalid"],
        ids=["layered", "discrete", "invalid"],
    )
    def test_backend_selection(self, datatype):
        """
        Test to check whether the correct backend is selected based on the 'datatype'
        attribute.

        """
        df = pd.DataFrame()
        df.datatype = datatype

        if datatype == "invalid":
            with pytest.raises(
                TypeError, match="No Data backend available for invalid"
            ):
                df.gstda
        else:
            backend = DATA[datatype]
            assert isinstance(df.gstda._backend, backend)
