from pathlib import Path
from typing import Any, Iterable, List, Literal, get_args

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import geometry as gmt

from geost import utils
from geost.abstract_classes import AbstractData
from geost.analysis import cumulative_thickness
from geost.export import (
    borehole_to_multiblock,
    export_to_dftgeodata,
    layerdata_to_pyvista_unstructured,
)

type Coordinate = int | float
type GeometryType = gmt.base.BaseGeometry | list[gmt.base.BaseGeometry]

HeaderType = Literal["point", "line"]
DataType = Literal["layered", "discrete"]


class LayeredData(AbstractData):
    """
    A class to hold layered data objects (i.e. containing "tops" and "bottoms") like
    borehole descriptions which can be used for selections and exports.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing the data. Mandatory columns that must be present in the
        DataFrame are: "nr", "x", "y", "surface", "top" and "bottom". Otherwise, many methods
        in the class will not work.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @staticmethod
    def _to_iterable(selection_values: str | Iterable) -> Iterable:
        if isinstance(selection_values, str):
            selection_values = [selection_values]
        return selection_values

    @staticmethod
    def _change_depth_values(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "top"] = df["surface"] - df["top"]
        df.loc[:, "bottom"] = df["surface"] - df["bottom"]
        return df

    def to_header(
        self,
        headertype: HeaderType = "point",
        horizontal_reference: str | int | CRS = 28992,
    ):
        """
        Generate a :class:`~geost.base.PointHeader` from this instance of LayeredData.

        Parameters
        ----------
        headertype : str, optional
            Type of header to generate. Must be one of "point" or "line". The default is
            "point".
        horizontal_reference : str | int | CRS, optional
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(), by default 28992.

        Returns
        -------
        :class:`~geost.base.PointHeader`
            An instance of :class:`~geost.base.PointHeader`

        """
        if headertype not in get_args(HeaderType):
            raise ValueError(
                f"Invalid headertype: {headertype}. Must be one of {get_args(HeaderType)}"
            )

        header_columns = ["nr", "x", "y", "surface", "end"]
        header = self._df[header_columns].drop_duplicates("nr").reset_index(drop=True)
        header = utils.dataframe_to_geodataframe(header).set_crs(horizontal_reference)
        header.headertype = headertype
        return header

    def to_collection(
        self,
        has_inclined: bool = False,
        horizontal_reference: str | int | CRS = 28992,
        vertical_reference: str | int | CRS = 5709,
        headertype: HeaderType = "point",
        datatype: DataType = "layered",
    ):
        """
        Create a collection from this instance of LayeredData. A collection combines
        header and data and ensures that they remain aligned when applying methods.

        Parameters
        ----------
        has_inclined : bool, optional
            If True, the data also contains inclined objects which means the top of layers
            is not in the same x,y-location as the bottom of layers. The default is False.
        horizontal_reference : str | int | CRS, optional
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(), by default 28992.
        vertical_reference : str | int | CRS, optional
            EPSG of the target vertical datum. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
            "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
            5710, by default 5709.
        headertype : HeaderType, optional
            Type of header to generate. Must be one of "point" or "line". The default is
            "point".
        datatype : DataType, optional
            Type of data to generate. Must be one of "layered" or "discrete". The default
            is "layered" which typically contains top and bottom depth information.

        Returns
        -------
        :class:`~geost.base.Collection`
            An instance of :class:`~geost.base.Collection`
        """
        from geost.base import BoreholeCollection  # Avoid circular import

        if datatype not in get_args(DataType):
            raise ValueError(
                f"Invalid datatype: {datatype}. Must be one of {get_args(DataType)}"
            )

        header = self.to_header(headertype, horizontal_reference)
        return BoreholeCollection(
            header,
            self._df,
            has_inclined=has_inclined,
            horizontal_reference=horizontal_reference,
            vertical_reference=vertical_reference,
        )
        # NOTE: Type of Collection may need to be inferred in the future.

    def select_by_values(
        self, column: str, selection_values: str | Iterable, how: str = "or"
    ):
        """
        Select data based on the presence of given values in a given column. Can be used
        for example to select boreholes that contain peat in the lithology column.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        selection_values : str | Iterable
            Value or values to look for in the column.
        how : str, optional
            Either "and" or "or". "and" requires all selection values to be present in
            column for selection. "or" will select the core if any one of the
            selection_values are found in the column. Default is "and".

        Returns
        -------
        :class:`~geost.base.LayeredData`
            New instance containing only the data selected by this method.

        Examples
        --------
        To select boreholes where both clay ("K") and peat ("V") are present at the same
        time, use "and" as a selection method:

        >>> boreholes.select_by_values("lith", ["V", "K"], how="and")

        To select boreholes that can have one, or both lithologies, use or as the selection
        method:

        >>> boreholes.select_by_values("lith", ["V", "K"], how="and")

        """
        if column not in self._df.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        selection_values = self._to_iterable(selection_values)

        selected = self._df
        if how == "or":
            valid = self._df["nr"][self._df[column].isin(selection_values)].unique()
            selected = selected[selected["nr"].isin(valid)]

        elif how == "and":
            for value in selection_values:
                valid = self._df["nr"][self._df[column] == value].unique()
                selected = selected[selected["nr"].isin(valid)]

        return selected

    def slice_depth_interval(
        self,
        upper_boundary: float | int = None,
        lower_boundary: float | int = None,
        relative_to_vertical_reference: bool = False,
        update_layer_boundaries: bool = True,
    ):
        """
        Slice data based on given upper and lower boundaries. This returns a new object
        containing only the sliced data.

        Parameters
        ----------
        upper_boundary : float | int, optional
            Every layer that starts above this is removed. The default is None.
        lower_boundary : float | int, optional
            Every layer that starts below this is removed. The default is None.
        relative_to_vertical_reference : bool, optional
            If True, the slicing is done with respect to any kind of vertical reference
            plane (e.g. "NAP", "TAW"). If False, the slice is done with respect to depth
            below the surface. The default is False.
        update_layer_boundaries : bool, optional
            If True, the layer boundaries in the sliced data are updated according to the
            upper and lower boundaries used with the slice. If False, the original layer
            boundaries are kept in the sliced object. The default is False.

        Returns
        -------
        :class:`~geost.base.LayeredData`
            New instance containing only the data selected by this method.

        Examples
        --------
        Usage depends on whether the slicing is done with respect to depth below the
        surface or to a vertical reference plane.

        For example, select layers in data that are between 2 and 3 meters below the
        surface:

        >>> data.slice_depth_interval(2, 3)

        By default, the method updates the layer boundaries in sliced object according to
        the upper and lower boundaries. To suppress this behaviour use:

        >>> data.slice_depth_interval(2, 3, update_layer_boundaries=False)

        Slicing can also be done with respect to a vertical reference plane like "NAP".
        For example, to select layers in boreholes that are between -3 and -5 m NAP, use:

        >>> data.slice_depth_interval(-3, -5, relative_to_vertical_reference=True)

        """
        if not upper_boundary:
            upper_boundary = 1e34 if relative_to_vertical_reference else -1e34

        if not lower_boundary:
            lower_boundary = -1e34 if relative_to_vertical_reference else 1e34

        sliced = self._df.copy()

        bounds_are_series = False
        if relative_to_vertical_reference:
            bounds_are_series = True
            upper_boundary = self._df["surface"] - upper_boundary
            lower_boundary = self._df["surface"] - lower_boundary

        sliced = sliced[
            (sliced["bottom"] > upper_boundary) & (sliced["top"] < lower_boundary)
        ]

        if update_layer_boundaries:
            if bounds_are_series:
                upper_boundary = upper_boundary.loc[sliced.index]
                lower_boundary = lower_boundary.loc[sliced.index]

            sliced.loc[sliced["top"] <= upper_boundary, "top"] = upper_boundary
            sliced.loc[sliced["bottom"] >= lower_boundary, "bottom"] = lower_boundary

        return sliced

    def slice_by_values(
        self, column: str, selection_values: str | Iterable, invert: bool = False
    ):
        """
        Slice rows from data based on matching condition. E.g. only return rows with
        a certain lithology in the collection object.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        selection_values : str | Iterable
            Values to look for in the column.
        invert : bool, optional
            If True, invert the slicing action, so remove layers with selected values
            instead of keeping them. The default is False.

        Returns
        -------
        :class:`~geost.base.LayeredData`
            New instance containing only the data objects selected by this method.

        Examples
        --------
        Return only rows in data contain sand ("Z") as lithology:

        >>> data.slice_by_values("lith", "Z")

        If you want all the rows that may contain everything but sand, use the "invert"
        option:

        >>> data.slice_by_values("lith", "Z", invert=True)

        """
        selection_values = self._to_iterable(selection_values)

        sliced = self._df.copy()

        if invert:
            sliced = sliced[~sliced[column].isin(selection_values)]
        else:
            sliced = sliced[sliced[column].isin(selection_values)]

        return sliced

    def select_by_condition(self, condition: Any, invert: bool = False):
        """
        Select data using a manual condition that results in a boolean mask. Returns the
        rows in the data where the 'condition' evaluates to True.

        Parameters
        ----------
        condition : list, pd.Series or array like
            Boolean array like object with locations at which the values will be
            preserved, dtype must be 'bool' and the length must correspond with the
            length of the data.
        invert : bool, optional
            If True, the selection is inverted so rows that evaluate to False will be
            returned. The default is False.

        Returns
        -------
        :class:`~geost.base.LayeredData`
            New instance containing only the data objects selected by this method.

        Examples
        --------
        Select rows in data that contain a specific value:

        >>> data.select_by_condition(data["lith"]=="V")

        Or select rows in the data that contain a specific (part of) string or strings:

        >>> data.select_by_condition(data["column"].str.contains("foo|bar"))

        """
        if invert:
            selected = self[~condition]
        else:
            selected = self[condition]
        return selected

    def get_cumulative_thickness(self, column: str, values: str | List[str]):
        """
        Get the cumulative thickness of layers where a column contains a specified search
        value or values.

        Parameters
        ----------
        column : str
            Name of column that must contain the search value or values.
        values : str | List[str]
            Search value or values in the column to find the cumulative thickness for.

        Returns
        -------
        pd.DataFrame
            Borehole ids and cumulative thickness of selected layers.

        Examples
        --------
        Get the cumulative thickness of the layers with lithology "K" in the column "lith"
        use:

        >>> data.get_cumulative_thickness("lith", "K")

        Or get the cumulative thickness for multiple selection values. In this case, a
        Pandas DataFrame is returned with a column per selection value containing the
        cumulative thicknesses:

        >>> data.get_cumulative_thickness("lith", ["K", "Z"])

        """
        selected_layers = self.slice_by_values(column, values)
        cum_thickness = selected_layers.groupby(["nr", column]).apply(
            cumulative_thickness, include_groups=False
        )
        cum_thickness = cum_thickness.unstack(level=column)
        return cum_thickness

    def get_layer_top(
        self,
        column: str,
        values: str | List[str],
        min_thickness: float = 0,
        min_depth: float = 0,
    ):
        """
        Find the depth at which a specified layer first occurs, starting at min_depth
        and looking downwards until the first layer of min_thickness is found of the
        specified layer.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data.
        values : str | List[str]
            Value or values of entries in the column that you want to find top of.
        min_thickness : float, optional
            Minimal thickness of the layer to be considered. The default is 0.
        min_depth : float, optional
            Minimal depth of the layer to be considered. The default is 0.

        Returns
        -------
        pd.DataFrame
            Borehole ids and top levels of selected layers in meters below the surface.

        Examples
        --------
        Get the top depth of layers in data where the lithology in the "lith" column
        is sand ("Z"):

        >>> data.get_layer_top("lith", "Z")

        """
        selected_layers = self.slice_by_values(column, values)
        selected_layers = selected_layers.slice_depth_interval(upper_boundary=min_depth)
        selected_layers = selected_layers.select_by_condition(
            selected_layers["bottom"] - selected_layers["top"] >= min_thickness
        )
        layer_top = selected_layers.groupby(["nr", column])["top"].first()
        return layer_top.unstack(level=column)

    def to_pyvista_cylinders(
        self,
        displayed_variables: str | List[str],
        radius: float = 1,
        vertical_factor: float = 1.0,
        relative_to_vertical_reference: bool = True,
    ):
        """
        Create a Pyvista MultiBlock object of cylinder-shaped geometries to represent
        boreholes. Although cylinders are prettier when visualized, they are quite costly
        to render in large numbers. Consider using
        :meth:`~geost.base.LayeredData.to_pyvista_grid` instead for large datasets.

        Parameters
        ----------
        displayed_variables : str | List[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius of the cylinders in m in the MultiBlock. The default is 1.
        vertical_factor : float, optional
            Factor to correct vertical scale. For example, when layer boundaries are given
            in cm, use 0.01 to convert to m. The default is 1.0, so no correction is applied.
            It is not recommended to use this for vertical exaggeration, use viewer functionality
            for that instead.
        relative_to_vertical_reference : bool, optional
            If True, the depth of the objects in the vtm file will be with respect to a
            reference plane (e.g. "NAP", "TAW"). If False, the depth will be with respect
            to 0.0. The default is True.

        Returns
        -------
        pyvista.MultiBlock
            A composite class holding the data which can be iterated over.

        """
        data_columns = self._to_iterable(displayed_variables)

        data = self._df.copy()

        if relative_to_vertical_reference:
            data = self._change_depth_values(data)
        else:
            data["surface"] = 0

        vtk_object = borehole_to_multiblock(data, data_columns, radius, vertical_factor)
        return vtk_object

    def to_pyvista_grid(
        self,
        displayed_variables: str | list[str],
        radius: float = 1,
    ):
        """
        Create a PyVista UnstructuredGrid object of the data in this instance. This
        method is more efficient than :meth:`~geost.base.LayeredData.to_pyvista_cylinders`
        for large datasets, as it uses a grid representation instead of cylinders.

        Parameters
        ----------
        data_columns : str | list[str]
            Name or names of data columns to include for visualisation. Can be columns that
            contain an array of floats, ints and strings.
        radius : float, optional
            Radius cells in m, by default 1.

        Returns
        -------
        pyvista.UnstructuredGrid
            A PyVista UnstructuredGrid object containing the data that can be used for
            3D visualisation in PyVista or other VTK viewers.

        """
        displayed_variables = self._to_iterable(displayed_variables)
        vtk_object = layerdata_to_pyvista_unstructured(
            self._df, displayed_variables, radius=radius
        )
        return vtk_object

    def to_datafusiontools(
        self,
        columns: List[str],
        outfile: str | Path = None,
        encode: bool = False,
        relative_to_vertical_reference: bool = True,
    ):
        """
        Export all data to the core "Data" class of Deltares DataFusionTools. Returns
        a list of "Data" objects, one for each data object that is exported. This list
        can directly be used within DataFusionTools. If out_file is given, the list of
        Data objects is saved to a pickle file.

        For DataFusionTools visit:
        https://bitbucket.org/DeltaresGEO/datafusiontools/src/master/

        Parameters
        ----------
        columns : List[str]
            Which columns in the data to include for the export. These will become variables
            in the DataFusionTools "Data" class.
        outfile : str | Path, optional
            If a path to outfile is given, the data is written to a pickle file.
        encode : bool, default True
            If True, categorical data columns are encoded to additional binary columns
            (all possible values become a seperate feature that is 0 or 1). The default is
            False. Warning: if there is a large number of possible categories, many columns
            with categorical data or both, the export process may become slow and may consume
            a large amount memory. Please consider carefully which categorical data columns
            need to be included.
        relative_to_vertical_reference : bool, optional
            If True, the depth of all data objects will converted to a depth with respect to
            a reference plane (e.g. "NAP", "TAW"). If False, the depth will be kept as original
            in the "top" and "bottom" columns which is in meter below the surface. The default
            is True.

        Returns
        -------
        List[Data]
            List containing the DataFusionTools Data objects.

        """
        columns = self._to_iterable(columns)
        data = self._df.copy()
        if relative_to_vertical_reference:
            data = self._change_depth_values(data)

        dftgeodata = export_to_dftgeodata(data, columns, encode=encode)

        if outfile:
            utils.save_pickle(dftgeodata, outfile)
        else:
            return dftgeodata

    def _create_geodataframe_3d(
        self,
        relative_to_vertical_reference: bool = True,
        crs: str | int | CRS = None,
        has_inclined: bool = False,
    ):
        """
        Helper method for export method "to_qgis3d" to create the necessary GeoDataFrame
        containing 3D Shapely objects and associated information.

        Parameters
        ----------
        relative_to_vertical_reference : bool, optional
            If True, the depth of all data objects will converted to a depth with respect to
            a reference plane (e.g. "NAP", "TAW"). If False, the depth will be kept as original
            in the "top" and "bottom" columns which is in meter below the surface. The default
            is True.
        crs : str | int | CRS
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input().
        has_inclined : bool, optional
            If True, the data also contains inclined objects which means the top of layers
            is not in the same x,y-location as the bottom of layers. The default is False.

        """
        data = self._df.copy()

        if relative_to_vertical_reference:
            data = self._change_depth_values(data)

        data_columns = [
            col
            for col in data.columns
            if col
            not in ["nr", "x", "y", "x_bot", "y_bot", "surface", "end", "top", "bottom"]
        ]

        data_to_write = dict(
            nr=data["nr"].values,
            top=data["top"].values.astype(float),
            bottom=data["bottom"].values.astype(float),
        )

        data_to_write.update(data[data_columns].to_dict(orient="list"))

        if has_inclined:
            geometries = [
                gmt.LineString([[x, y, top + 0.01], [x_bot, y_bot, bottom + 0.01]])
                for x, y, x_bot, y_bot, top, bottom in zip(
                    data["x"].values.astype(float),
                    data["y"].values.astype(float),
                    data["x_bot"].values.astype(float),
                    data["y_bot"].values.astype(float),
                    data["top"].values.astype(float),
                    data["bottom"].values.astype(float),
                )
            ]
        else:  # NOTE: Doesn't it need to be "top - 0.01" to create overlap?
            geometries = [
                gmt.LineString([[x, y, top + 0.01], [x, y, bottom + 0.01]])
                for x, y, top, bottom in zip(
                    data["x"].values.astype(float),
                    data["y"].values.astype(float),
                    data["top"].values.astype(float),
                    data["bottom"].values.astype(float),
                )
            ]

        gdf = gpd.GeoDataFrame(
            data=data_to_write,
            geometry=geometries,
            crs=crs,
        )
        return gdf

    def to_qgis3d(
        self,
        outfile: str | Path,
        relative_to_vertical_reference: bool = True,
        crs: str | int | CRS = None,
        has_inclined: bool = False,
        **kwargs,
    ):
        """
        Write data to geopackage file that can be directly loaded in the Qgis2threejs
        plugin. Works only for layered (borehole) data.

        Parameters
        ----------
        outfile : str | Path
            Path to geopackage file to be written.
        relative_to_vertical_reference : bool, optional
            If True, the depth of all data objects will converted to a depth with
            respect to a reference plane (e.g. "NAP", "Ostend height"). If False, the
            depth will be kept as original in the "top" and "bottom" columns which is in
            meter below the surface. The default is True.
        crs : str | int | CRS
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input().
        has_inclined : bool, optional
            If True, the data also contains inclined objects which means the top of layers
            is not in the same x,y-location as the bottom of layers. The default is False.

        **kwargs
            geopandas.GeodataFrame.to_file kwargs. See relevant Geopandas documentation.

        """
        qgis3d = self._create_geodataframe_3d(
            relative_to_vertical_reference, crs=crs, has_inclined=has_inclined
        )
        qgis3d.to_file(outfile, driver="GPKG", **kwargs)

    def to_kingdom(
        self,
        outfile: str | Path,
        tdstart: int = 1,
        vw: float = 1500.0,
        vs: float = 1600.0,
    ):
        """
        Write data to 2 csv files: 1) interval data and 2) time-depth chart. These files
        can be imported in the Kingdom seismic interpretation software.

        Parameters
        ----------
        outfile : str | Path
            Path to csv file to be written.
        tdstart : int
            startindex for TDchart, default is 1
        vw : float
            sound velocity in water in m/s, default is 1500 m/s
        vs : float
            sound velocity in sediment in m/s, default is 1600 m/s
        """
        # 1. add column needed in kingdom and write interval data
        kingdom_df = self._df.copy()
        # Add total depth and rename bottom and top columns to Kingdom requirements
        kingdom_df.insert(7, "Total depth", (kingdom_df["surface"] - kingdom_df["end"]))
        kingdom_df.rename(
            columns={"top": "Start depth", "bottom": "End depth"}, inplace=True
        )
        kingdom_df.to_csv(outfile, index=False)

        # 2. create and write time-depth chart
        tdchart = self._df[["nr", "surface"]].copy()
        tdchart.drop_duplicates(inplace=True)
        tdchart.insert(0, "id", range(tdstart, tdstart + len(tdchart)))
        # Add measured depth (predefined depths of 0 and 1 m below surface)
        tdchart = pd.concat(
            [
                tdchart.assign(MD=np.zeros(len(tdchart), dtype=np.int64)),
                tdchart.assign(MD=np.ones(len(tdchart), dtype=np.int64)),
            ]
        )
        # Add two-way travel time
        tdchart["TWT"] = (-tdchart["surface"] / (vw / 2 / 1000)) + (
            tdchart["MD"] * 1 / (vs / 2 / 1000)
        )

        tdchart.drop("surface", axis=1, inplace=True)
        tdchart.sort_values(by=["id", "MD"], inplace=True)
        if not isinstance(outfile, Path):
            outfile = Path(outfile)
        tdchart.to_csv(
            outfile.parent.joinpath(f"{outfile.stem}_TDCHART{outfile.suffix}"),
            index=False,
        )


class DiscreteData(AbstractData):
    def __init__(self, df):
        self._df = df

    @staticmethod
    def _to_iterable(selection_values: str | Iterable) -> Iterable:
        if isinstance(selection_values, str):
            selection_values = [selection_values]
        return selection_values

    def to_header(
        self,
        horizontal_reference: str | int | CRS = 28992,
        vertical_reference: str | int | CRS = 5709,
    ):
        """
        Generate a :class:`~geost.base.PointHeader` from this instance of DiscreteData.

        Parameters
        ----------
        horizontal_reference : str | int | CRS, optional
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(), by default 28992.
        vertical_reference : str | int | CRS, optional
            EPSG of the target vertical datum. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
            "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
            5710, by default 5709.

        Returns
        -------
        :class:`~geost.base.PointHeader`
            An instance of :class:`~geost.base.PointHeader`

        """
        from geost.accessors.header import PointHeader

        header_columns = ["nr", "x", "y", "surface", "end"]
        header = self._df[header_columns].drop_duplicates("nr").reset_index(drop=True)
        header = utils.dataframe_to_geodataframe(header).set_crs(horizontal_reference)
        return PointHeader(header, vertical_reference)

    def to_collection(
        self,
        horizontal_reference: str | int | CRS = 28992,
        vertical_reference: str | int | CRS = 5709,
    ):
        """
        Create a collection from this instance of DiscreteData. A collection combines
        header and data and ensures that they remain aligned when applying methods.

        Parameters
        ----------
        horizontal_reference : str | int | CRS, optional
            EPSG of the target crs. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(), by default 28992.
        vertical_reference : str | int | CRS, optional
            EPSG of the target vertical datum. Takes anything that can be interpreted by
            pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
            "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
            5710, by default 5709.

        Returns
        -------
        :class:`~geost.base.Collection`
            An instance of :class:`~geost.base.Collection`
        """
        from geost.base import CptCollection

        header = self.to_header(horizontal_reference, vertical_reference)
        return CptCollection(
            header, self._df
        )  # TODO: type of Collection needs to be inferred in the future

    def select_by_values(
        self, column: str, selection_values: str | Iterable, how: str = "or"
    ):
        """
        Select data based on the presence of given values in a given column containing
        categorical data. Can be used for example to select points that contain peat in
        the lithology column.

        Parameters
        ----------
        column : str
            Name of column that contains categorical data to use when looking for
            values.
        selection_values : str | Iterable
            Value or values to look for in the column.
        how : str, optional
            Either "and" or "or". "and" requires all selection values to be present in
            column for selection. "or" will select the core if any one of the
            selection_values are found in the column. Default is "and".

        Returns
        -------
        :class:`~geost.base.DiscreteData`
            New instance containing only the data selected by this method.

        Examples
        --------
        To select data where both clay ("K") and peat ("V") are present at the same time,
        use "and" as a selection method:

        >>> data.select_by_values("lith", ["V", "K"], how="and")

        To select data that can have one, or both lithologies, use or as the selection
        method:

        >>> data.select_by_values("lith", ["V", "K"], how="and")

        """
        if column not in self._df.columns:
            raise IndexError(
                f"The column '{column}' does not exist and cannot be used for selection"
            )

        selection_values = self._to_iterable(selection_values)

        selected = self._df.copy()
        if how == "or":
            valid = self._df["nr"][self._df[column].isin(selection_values)].unique()
            selected = selected[selected["nr"].isin(valid)]

        elif how == "and":
            for value in selection_values:
                valid = self._df["nr"][self._df[column] == value].unique()
                selected = selected[selected["nr"].isin(valid)]

        return selected

    def slice_depth_interval(
        self,
        upper_boundary: float | int = None,
        lower_boundary: float | int = None,
        relative_to_vertical_reference: bool = False,
    ):
        """
        Slice data based on given upper and lower boundaries. This returns a new object
        containing only the sliced data.

        Parameters
        ----------
        upper_boundary : float | int, optional
            Every layer that starts above this is removed. The default is None.
        lower_boundary : float | int, optional
            Every layer that starts below this is removed. The default is None.
        relative_to_vertical_reference : bool, optional
            If True, the slicing is done with respect to any kind of vertical reference
            plane (e.g. "NAP", "TAW"). If False, the slice is done with respect to depth
            below the surface. The default is False.
        update_layer_boundaries : bool, optional
            If True, the layer boundaries in the sliced data are updated according to the
            upper and lower boundaries used with the slice. If False, the original layer
            boundaries are kept in the sliced object. The default is False.

        Returns
        -------
        :class:`~geost.base.DiscreteData`
            New instance containing only the data selected by this method.

        Examples
        --------
        Usage depends on whether the slicing is done with respect to depth below the
        surface or to a vertical reference plane.

        For example, select layers in data that are between 2 and 3 meters below the
        surface:

        >>> data.slice_depth_interval(2, 3)

        Slicing can also be done with respect to a vertical reference plane like "NAP".
        For example, to select layers in boreholes that are between -3 and -5 m NAP, use:

        >>> data.slice_depth_interval(-3, -5, relative_to_vertical_reference=True)

        """
        if not upper_boundary:
            upper_boundary = 1e34 if relative_to_vertical_reference else -1e34

        if not lower_boundary:
            lower_boundary = -1e34 if relative_to_vertical_reference else 1e34

        sliced = self._df.copy()

        if relative_to_vertical_reference:
            upper_boundary = self._df["surface"] - upper_boundary
            lower_boundary = self._df["surface"] - lower_boundary

        sliced = sliced[
            (sliced["depth"] >= upper_boundary) & (sliced["depth"] <= lower_boundary)
        ]

        return sliced

    def slice_by_values(self):  # pragma: no cover
        raise NotImplementedError()

    def select_by_condition(self, condition: Any, invert: bool = False):
        """
        Select data using a manual condition that results in a boolean mask. Returns the
        rows in the data where the 'condition' evaluates to True.

        Parameters
        ----------
        condition : list, pd.Series or array like
            Boolean array like object with locations at which the values will be
            preserved, dtype must be 'bool' and the length must correspond with the
            length of the data.
        invert : bool, optional
            If True, the selection is inverted so rows that evaluate to False will be
            returned. The default is False.

        Returns
        -------
        :class:`~geost.base.DiscreteData`
            New instance containing only the data objects selected by this method.

        Examples
        --------
        Select rows in data where column values are larger than:

        >>> data.select_by_condition(data["column"] > 2)

        Or select rows in the data based on multiple conditions:

        >>> data.select_by_condition((data["column1"] > 2) & (data["column2] < 1))

        """
        if invert:
            selected = self._df[~condition]
        else:
            selected = self._df[condition]
        return selected

    def get_cumulative_thickness(self):  # pragma: no cover
        raise NotImplementedError()

    def get_layer_top(self):  # pragma: no cover
        raise NotImplementedError()

    def to_pyvista_cylinders(self):  # pragma: no cover
        raise NotImplementedError()

    def to_pyvista_grid(self):  # pragma: no cover
        raise NotImplementedError()

    def to_datafusiontools(self):  # pragma: no cover
        raise NotImplementedError()
