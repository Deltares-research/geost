import pandas as pd
import pytest

from geost.accessors.accessor import DATA, HEADER


class TestHeaderAccessor:
    @pytest.mark.unittest
    def test_validate(self):
        """
        Test to check wheter an input DataFrame has the required 'headertype' attribute
        which is needed to choose the correct backend for the accessor.

        """
        df = pd.DataFrame()
        with pytest.raises(
            AttributeError,
            match="Header has no attribute 'headertype', gsthd accessor cannot choose backend",
        ):
            df.gsthd  # Triggers `self._validate(gdf)` in `Header.__init__`

    @pytest.mark.parametrize(
        "headertype",
        ["point", "line", "invalid"],
        ids=["point", "line", "invalid"],
    )
    def test_backend_selection(self, headertype):
        """
        Test to check whether the correct backend is selected based on the 'headertype'
        attribute.

        """
        df = pd.DataFrame()
        df.headertype = headertype

        if headertype == "invalid":
            with pytest.raises(
                TypeError, match="No Header backend available for invalid"
            ):
                df.gsthd
        else:
            backend = HEADER[headertype]
            assert isinstance(df.gsthd._backend, backend)


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
