import geopandas as gpd
import pandas as pd

from . import data, header

HEADER_BACKEND = {
    "point": header.PointHeader,
    "linestring": header.LineHeader,
}
DATA_BACKEND = {
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
        header = HEADER_BACKEND.get(geom_type)
        if header is None:
            raise TypeError(f"No Header backend available for {geom_type}")
        return header(self._gdf)


@pd.api.extensions.register_dataframe_accessor("gstda")
class Data:
    def __init__(self, df):
        self._df = df
        self._backend = self._get_backend()

    def __getattr__(self, attr):
        return getattr(self._backend, attr)

    def _get_backend(self):
        if {"top", "bottom"}.issubset(self._df.columns):
            backend = DATA_BACKEND["layered"]
        elif "depth" in self._df.columns:
            backend = DATA_BACKEND["discrete"]
        else:
            raise KeyError(
                "No 'top' and 'bottom' or 'depth' columns present. Data accessor cannot "
                "determine 'layered' or 'discrete' backend."
            )
        return backend(self._df)
