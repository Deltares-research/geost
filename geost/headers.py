from abc import ABC, abstractmethod
from pathlib import WindowsPath
from typing import Iterable

import geopandas as gpd

from geost import spatial
from geost.mixins import GeopandasExportMixin
from geost.validate.decorators import validate_header

type Coordinate = int | float


class AbstractHeader(ABC):
    @property
    @abstractmethod
    def gdf(self):
        pass

    @gdf.setter
    @abstractmethod
    def gdf(self, gdf):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def select_within_bbox(self):
        pass

    @abstractmethod
    def select_with_points(self):
        pass

    @abstractmethod
    def select_with_lines(self):
        pass

    @abstractmethod
    def select_within_polygons(self):
        pass

    @abstractmethod
    def select_by_depth(self):
        pass

    @abstractmethod
    def select_by_length(self):
        pass

    @abstractmethod
    def get_area_labels(self):
        pass


class PointHeader(AbstractHeader, GeopandasExportMixin):
    def __init__(self, gdf):
        self.gdf = gdf

    def __repr__(self):
        return f"{self.__class__.__name__} instance containing {len(self.gdf)} objects"

    @property
    def gdf(self):
        return self._gdf

    @gdf.setter
    @validate_header
    def gdf(self, gdf):
        self._gdf = gdf

    def get(self, selection_values: str | Iterable, column: str = "nr"):
        """
        Get a subset of a collection through a string or iterable of object id(s).
        Optionally uses a different column than "nr" (the column with object ids).

        Parameters
        ----------
        selection_values : str | Iterable
            Values to select.
        column : str, optional
            In which column of the header to look for selection values, by default "nr".

        Returns
        -------
        Instance of :class:`~geost.headers.PointHeader`.
            Instance of :class:`~geost.headers.PointHeader` containing only
            objects selected through this method.

        Examples
        --------
        self.get(["obj1", "obj2"]) will return a collection with only these objects.

        Suppose we have a collection of boreholes that we have joined with geological
        map units using the method
        :meth:`~geost.headers.PointHeader.get_area_labels`. We have added this data
        to the header table in the column 'geological_unit'. Using:

        self.get(["unit1", "unit2"], column="geological_unit")

        will return a :class:`~geost.headers.PointHeader` with all boreholes
        that are located in "unit1" and "unit2" geological map areas.
        """
        if isinstance(selection_values, str):
            selected_gdf = self.gdf[self.gdf[column] == selection_values]
        elif isinstance(selection_values, Iterable):
            selected_gdf = self.gdf[self.gdf[column].isin(selection_values)]

        selected_gdf = selected_gdf[~selected_gdf.duplicated()]
        # selection = self.data.loc[self.data["nr"].isin(selected_header["nr"])]

        return self.__class__(selected_gdf)

    def select_within_bbox(
        self,
        xmin: Coordinate,
        xmax: Coordinate,
        ymin: Coordinate,
        ymax: Coordinate,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on a bounding box.

        Parameters
        ----------
        xmin : float | int
            Minimum x-coordinate of the bounding box.
        xmax : float | int
            Maximum x-coordinate of the bounding box.
        ymin : float | int
            Minimum y-coordinate of the bounding box.
        ymax : float | int
            Maximum y-coordinate of the bounding box.
        invert : bool, optional
            Invert the selection, so select all objects outside of the
            bounding box in this case, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.gdf_from_bbox(
            self.gdf, xmin, xmax, ymin, ymax, invert=invert
        )
        return self.__class__(gdf_selected)

    def select_with_points(
        self,
        points: str | WindowsPath | gpd.GeoDataFrame,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on point geometries.

        Parameters
        ----------
        points : str | WindowsPath | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.
        buffer : float | int
            Buffer distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.gdf_from_points(self.gdf, points, buffer, invert=invert)
        return self.__class__(gdf_selected)

    def select_with_lines(
        self,
        lines: str | WindowsPath | gpd.GeoDataFrame,
        buffer: float | int,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on line geometries.

        Parameters
        ----------
        lines : str | WindowsPath | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.
        buffer : float | int
            Buffer distance for selection geometries.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.gdf_from_lines(self.gdf, lines, buffer, invert=invert)
        return self.__class__(gdf_selected)

    def select_within_polygons(
        self,
        polygons: str | WindowsPath | gpd.GeoDataFrame,
        buffer: float | int = 0,
        invert: bool = False,
    ):
        """
        Make a selection of the header based on polygon geometries.

        Parameters
        ----------
        polygons : str | WindowsPath | gpd.GeoDataFrame
            Geodataframe (or file that can be parsed to a geodataframe) to select with.
        buffer : float | int, optional
            Optional buffer distance around the polygon selection geometries, by default
            0.
        invert : bool, optional
            Invert the selection, by default False.

        Returns
        -------
        :class:`~geost.headers.PointHeader`
            Instance of :class:`~geost.headers.PointHeader`containing only selected
            geometries.
        """
        gdf_selected = spatial.gdf_from_polygons(
            self.gdf, polygons, buffer, invert=invert
        )
        return self.__class__(gdf_selected)

    def select_by_depth(self):
        raise NotImplementedError("Add function logic")

    def select_by_length(self):
        raise NotImplementedError("Add function logic")

    def get_area_labels(self):
        raise NotImplementedError("Add function logic")

    def to_shape(self):
        raise NotImplementedError("Add function logic")

    def to_geoparquet(self):
        raise NotImplementedError("Add function logic")


class LineHeader(AbstractHeader, GeopandasExportMixin):
    def __init__(self, gdf):
        self.gdf = gdf

    def __repr__(self):
        return f"{self.__class__.__name__} instance containing {len(self.gdf)} objects"

    @property
    def gdf(self):
        return self._gdf

    @gdf.setter
    @validate_header
    def gdf(self, gdf):
        self._gdf = gdf

    def get(self):
        raise NotImplementedError("Add function logic")

    def select_within_bbox(self):
        raise NotImplementedError("Add function logic")

    def select_with_points(self):
        raise NotImplementedError("Add function logic")

    def select_with_lines(self):
        raise NotImplementedError("Add function logic")

    def select_within_polygons(self):
        raise NotImplementedError("Add function logic")

    def select_by_depth(self):
        raise NotImplementedError("Add function logic")

    def select_by_length(self):
        raise NotImplementedError("Add function logic")

    def get_area_labels(self):
        raise NotImplementedError("Add function logic")


class HeaderFactory:
    def __init__(self):
        self._types = {}

    def register_header(self, type_, header):
        self._types[type_] = header

    @staticmethod
    def get_geometry_type(gdf):
        geometry_type = gdf.geom_type.unique()
        if len(geometry_type) > 1:
            return "Multiple geometries"
        else:
            return geometry_type[0]

    def create_from(self, gdf, **kwargs):
        geometry_type = self.get_geometry_type(gdf)
        header = self._types.get(geometry_type)
        if not header:
            raise ValueError(f"No Header type available for {geometry_type}")
        return header(gdf, **kwargs)


header_factory = HeaderFactory()
header_factory.register_header("Point", PointHeader)
header_factory.register_header("Line", LineHeader)


if __name__ == "__main__":

    gdf_test = gpd.read_parquet(
        r"c:\Users\onselen\Development\experimenteel\drgi_wadden.geoparquet"
    )
    pheader = PointHeader(gdf_test)
    pheader.select_within_bbox(235000, 245000, 610000, 620000)
    print(PointHeader(2))
    print(LineHeader(2))
