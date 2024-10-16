from pathlib import Path

import geopandas as gpd
import pandas as pd


class PandasExportMixin:
    def to_csv(self, file: str | Path, **kwargs):
        """
        Write data to csv file.

        Parameters
        ----------
        file : str | Path
            Path to csv file to be written.
        **kwargs
            pd.DataFrame.to_csv kwargs. See relevant Pandas documentation.
        """
        self.df.to_csv(file, **kwargs)

    def to_parquet(self, file: str | Path, **kwargs):
        """
        Write data to parquet file.

        Parameters
        ----------
        file : str | Path
            Path to parquet file to be written.
        **kwargs
            pd.DataFrame.to_parquet kwargs. See relevant Pandas documentation.
        """
        self.df.to_parquet(file, **kwargs)


class GeopandasExportMixin:
    def to_csv(self, file: str | Path, **kwargs):
        """
        Write header to csv file as plain table (no geometries).

        Parameters
        ----------
        file : str | Path
            Path to csv file to be written.
        **kwargs
            pd.DataFrame.to_csv kwargs. See relevant Pandas documentation.
        """
        df = pd.DataFrame(self.gdf.drop(columns="geometry"))
        df.to_csv(file, **kwargs)

    def to_parquet(self, file: str | Path, **kwargs):
        """
        Write header to parquet file as plain table (no geometries).

        Parameters
        ----------
        file : str | Path
            Path to parquet file to be written.
        **kwargs
            pd.DataFrame.to_parquet kwargs. See relevant Pandas documentation.
        """
        df = pd.DataFrame(self.gdf.drop(columns="geometry"))
        df.to_parquet(file, **kwargs)

    def to_shape(self, file: str | Path, **kwargs):
        """
        Write header data to shapefile. You can use the resulting file to display
        borehole locations in GIS for instance.

        Parameters
        ----------
        file : str | Path
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_file kwargs. See relevant GeoPandas documentation.
        """
        self.gdf.to_file(file, **kwargs)

    def to_geopackage(self, file: str | Path, **kwargs):
        """
        Write header data to geopackage. You can use the resulting file to display
        borehole locations in GIS for instance.

        Parameters
        ----------
        file : str | Path
            Path to geopackage to be written.
        **kwargs
            gpd.GeoDataFrame.to_file kwargs. See relevant GeoPandas documentation.
        """
        self.gdf.to_file(file, **kwargs)

    def to_geoparquet(self, file: str | Path, **kwargs):
        """
        Write header data to geoparquet. You can use the resulting file to display
        borehole locations in GIS for instance. Please note that Geoparquet is supported
        by GDAL >= 3.5. For Qgis this means QGis >= 3.26

        Parameters
        ----------
        file : str | Path
            Path to shapefile to be written.
        **kwargs
            gpd.GeoDataFrame.to_parquet kwargs. See relevant Pandas documentation.
        """
        self.gdf.to_parquet(file, **kwargs)
