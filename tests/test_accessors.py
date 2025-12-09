import geopandas as gpd
import pandas as pd
import pytest

from geost.accessors.accessor import DATA_BACKEND, HEADER_BACKEND


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
            backend = HEADER_BACKEND[header]
            header = request.getfixturevalue(header)
            assert isinstance(header.gsthd._backend, backend)


class TestDataAccessor:
    @pytest.fixture
    def layered(self):
        df = pd.DataFrame({"top": [0, 30], "bottom": [30, 40]})
        return df

    @pytest.fixture
    def discrete(self):
        df = pd.DataFrame({"depth": [1, 2, 3]})
        return df

    @pytest.fixture
    def invalid(self):
        return (
            pd.DataFrame()
        )  # No "top", "bottom", or "depth" columns are invalid for the accessor.

    @pytest.mark.parametrize(
        "datatype",
        ["layered", "discrete", "invalid"],
        ids=["layered", "discrete", "invalid"],
    )
    def test_backend_selection(self, datatype, request):
        """
        Test to check whether the correct backend is selected based on the 'datatype'
        attribute.

        """
        df = request.getfixturevalue(datatype)
        if datatype == "invalid":
            expected_error = (
                "No 'top' and 'bottom' or 'depth' columns present. Data accessor cannot "
                "determine 'layered' or 'discrete' backend."
            )
            with pytest.raises(KeyError, match=expected_error):
                df.gstda
        else:
            backend = DATA_BACKEND[datatype]
            assert isinstance(df.gstda._backend, backend)
