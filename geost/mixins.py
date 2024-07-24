import geopandas as gpd
import pandas as pd


class PandasExportMixin:
    def to_csv(self):
        pass

    def to_parquet(self):
        pass


class GeopandasExportMixin:
    def to_shape(self):
        pass

    def to_geoparquet(self):
        pass
