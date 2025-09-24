import geopandas as gpd
import pandas as pd

from . import data, header

HEADER = {
    "point": header.PointHeader,
    "linestring": header.LineHeader,
}
DATA = {
    "layered": data.LayeredData,
    "discrete": data.DiscreteData,
}


@pd.api.extensions.register_dataframe_accessor("gsthd")
class Header:
    def __init__(self, gdf):
        self._gdf = gdf
        self._backend = self._get_backend()

    def __getattr__(self, attr):
        return getattr(self._backend, attr)

    @staticmethod
    def _get_geom_type(gdf):
        if isinstance(gdf, gpd.GeoDataFrame):
            geom = gdf.geom_type.iloc[0]
        else:
            raise TypeError("Header accessor only accepts GeoDataFrames.")
        return geom.lower()

    def _get_backend(self):
        geom_type = self._get_geom_type(self._gdf)
        header = HEADER.get(geom_type)
        if header is None:
            raise TypeError(f"No Header backend available for {geom_type}")
        return header(self._gdf)


@pd.api.extensions.register_dataframe_accessor("gstda")
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
