import pandas as pd

from . import data, header

HEADER = {
    "point": header.PointHeader,
    "line": header.LineHeader,
}
DATA = {
    "layered": data.LayeredData,
    "discrete": data.DiscreteData,
}


@pd.api.extensions.register_dataframe_accessor("geoheader")
class Header:
    def __init__(self, gdf):
        self._validate(gdf)
        self._gdf = gdf
        self._backend = self._get_backend()

    def __getattr__(self, attr):
        return getattr(self._backend, attr)

    @staticmethod
    def _validate(gdf):
        if not hasattr(gdf, "headertype"):
            raise AttributeError(
                "Header has no attribute 'headertype', geoheader accessor cannot choose backend"
            )

    def _get_backend(self):
        header = HEADER.get(self._gdf.headertype)
        if header is None:
            raise TypeError(f"No Header backend available for {self._gdf.headertype}")
        return header(self._gdf)


@pd.api.extensions.register_dataframe_accessor("geodata")
class Data:
    def __init__(self, df):
        self._validate(df)
        self._df = df
        self._backend = self._get_backend()

    def __getattr__(self, attr):
        return getattr(self._backend, attr)

    @staticmethod
    def _validate(df):
        if not hasattr(df, "datatype"):
            raise AttributeError(
                "Data has no attribute 'datatype', geodata accessor cannot choose backend"
            )

    def _get_backend(self):
        data = DATA.get(self._df.datatype)
        if data is None:
            raise TypeError(f"No Data backend available for {self._df.datatype}")
        return data(self._df)
