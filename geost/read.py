from pathlib import Path, WindowsPath
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from geost.bro import BroApi
from geost.io import _parse_cpt_gef_files
from geost.io.parsers import SoilCore
from geost.new_base import BoreholeCollection, CptCollection, LayeredData
from geost.utils import dataframe_to_geodataframe

geometry_to_selection_function = {
    "Polygon": "select_within_polygons",
    "Line": "select_with_lines",
    "Point": "select_with_points",
}


def __read_parquet(file: WindowsPath) -> pd.DataFrame:
    """
    Read parquet file.

    Parameters
    ----------
    file : WindowsPath
        Path to file to be read.

    Returns
    -------
    pd.DataFrame
        Dataframe with contents of 'file'.

    Raises
    ------
    TypeError
        if 'file' has no '.parquet' or '.pq' suffix.

    """
    suffix = file.suffix
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file)
    else:
        raise TypeError(
            f"Expected parquet file (with .parquet or .pq suffix) but got {suffix} file"
        )


def adjust_z_coordinates(data_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Interpret an array containing elevation or z-coordinates. GeoST data objects require
    that elevation to indicate layer top/bottom or discrete data point z-coordinates to
    be increasing downward while starting at 0. This function detects from a given array
    how elevation is defined and turns it into the required format if required.

    Parameters
    ----------
    data_dataframe : pd.DataFrame
        Dataframe with layer or discrete data that includes columns referecing layer
        top, bottoms or discrete data z-coordinates.
    """
    top_column_label = [col for col in data_dataframe.columns if col in ["top", "z"]][0]
    has_bottom_column = any([True for col in data_dataframe.columns if col == "bottom"])

    # TODO: think about detection. Only considers first two indices to determine
    # downward decreasing or increasing of z-coordinates.
    if data_dataframe[top_column_label].iloc[0] == data_dataframe["surface"].iloc[0]:
        data_dataframe[top_column_label] -= data_dataframe["surface"]
        if has_bottom_column:
            data_dataframe["bottom"] -= data_dataframe["surface"]
    if (
        data_dataframe[top_column_label].iloc[0]
        > data_dataframe[top_column_label].iloc[1]
    ):
        data_dataframe[top_column_label] *= -1
        if has_bottom_column:
            data_dataframe["bottom"] *= -1

    return data_dataframe


def read_sst_cores(
    file: str | WindowsPath,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
    has_inclined: bool = False,
    column_mapper: dict = {},
) -> BoreholeCollection:
    """
    Read Subsurface Toolbox native parquet file with core information. Such a file
    includes row data for each (borehole) layer and must at least include the following
    column headers:

    nr      - Object id
    x       - X-coordinates according to the given horizontal_reference
    y       - Y-coordinates according to the given horizontal_reference
    surface - Surface elevation according to the given vertical_reference
    end     - End depth according to the given vertical_reference
    top     - Top depth of each layer
    bottom  - Bottom depth of each layer

    If you are reading a file that uses different column names, see the optional
    argument "column_mapper" below. There are no further limits or requirements to
    additional (data) columns.

    Parameters
    ----------
    file : str | WindowsPath
        Path to file to be read.
    horizontal_reference (int): Horizontal reference, see
        EPSG of the data's coordinate reference system. Takes anything that can be
        interpreted by pyproj.crs.CRS.from_user_input().
    vertical_reference: str | int | CRS
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710.
    column_mapper: dict
        If the file to be read uses different column names than the ones given in the
        description above, you can use a dictionary mapping to translate the column
        names to the required format.

    Returns
    -------
    :class:`~geost.borehole.BoreholeCollection`
        Instance of :class:`~geost.borehole.BoreholeCollection`.

    Examples
    --------
    Suppose we have a file of boreholes with columns 'nr', 'x', 'y', 'maaiveld', 'end',
    'top' and 'bottom'. . The column 'maaiveld' corresponds to the required column
    'surface' in GeoST and thus needs to be remapped. The x and y-coordinates are given
    in WGS84 UTM 31N whereas the vertical datum is given in the Belgian Ostend height
    vertical datum:

    >>> read_sst_cores(file, 32631, 'Ostend height', column_mapper={'maaiveld': 'surface'})
    """
    column_mapper = {value: key for key, value in column_mapper.items()}
    sst_cores = __read_parquet(Path(file))
    sst_cores.rename(columns=column_mapper, inplace=True)

    # Figure out top and bottom reference and coerce to downward increasing from 0 if
    # required.
    sst_cores = adjust_z_coordinates(sst_cores)
    layerdata = LayeredData(sst_cores, has_inclined=has_inclined)
    collection = layerdata.to_collection(horizontal_reference, vertical_reference)
    return collection


def read_sst_cpts(
    file: str | WindowsPath,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
) -> CptCollection:
    """
    Read Subsurface Toolbox native parquet file with cpt information.

    Parameters
    ----------
    file : str | WindowsPath
        Path to file to be read.
    vertical_reference: str
        Which vertical reference is used for tops and bottoms. See
        :py:attr:`~geost.base.PointDataCollection.vertical_reference` for documentation
        of this attribute.
    horizontal_reference (int): Horizontal reference, see
        :py:attr:`~geost.base.PointDataCollection.horizontal_reference`

    Returns
    -------
    :class:`~geost.borehole.CptCollection`
        Instance of :class:`~geost.borehole.CptCollection`.
    """
    filepath = Path(file)
    sst_cpts = __read_parquet(filepath)
    return CptCollection(
        sst_cpts,
        vertical_reference=vertical_reference,
        horizontal_reference=horizontal_reference,
    )


def read_nlog_cores(file: str | WindowsPath) -> BoreholeCollection:
    """
    Read NLog boreholes from the 'nlog_stratstelsel' Excel file. You can find this
    distribution of borehole data here: https://www.nlog.nl/boringen

    Warning: reading this Excel file is really slow (~10 minutes). Converting it to
    parquet using :func:`~geost.utils.excel_to_parquet` once and using that file instead
    allows for much faster reading of nlog data.

    Parameters
    ----------
    file : str | WindowsPath
        Path to nlog_stratstelsel.xlsx or .parquet

    Returns
    -------
    :class:`~geost.borehole.BoreholeCollection`
        :class:`~geost.borehole.BoreholeCollection`
    """
    filepath = Path(file)
    if filepath.suffix == ".xlsx":
        nlog_cores = pd.read_excel(filepath)
    else:
        nlog_cores = __read_parquet(filepath)

    nlog_cores.rename(
        columns={
            "NITG_NR": "nr",
            "X_TOP_RD": "x",
            "Y_TOP_RD": "y",
            "X_BOTTOM_RD": "x_bot",
            "Y_BOTTOM_RD": "y_bot",
            "TV_TOP_NAP": "top",
            "TV_BOTTOM_NAP": "bottom",
        },
        inplace=True,
    )

    nlog_cores["top"] *= -1
    nlog_cores["bottom"] *= -1

    nrs = []
    x = []
    y = []
    x_bot = []
    y_bot = []
    surface = []
    end = []
    nrs_data = []
    surface_data = []
    end_data = []
    for nr, data in nlog_cores.groupby("nr"):
        nrs.append(nr)
        x.append(data["x"].iloc[0])
        y.append(data["y"].iloc[0])
        x_bot.append(data["x_bot"].iloc[0])
        y_bot.append(data["y_bot"].iloc[0])
        surface.append(data["top"].iloc[0])
        end.append(data["bottom"].iloc[-1])
        nrs_data += [nr for i in range(len(data))]
        surface_data += [data["top"].iloc[0] for i in range(len(data))]
        end_data += [data["bottom"].iloc[-1] for i in range(len(data))]

    surface_end_df = pd.DataFrame(
        {"nr": nrs_data, "surface": surface_data, "end": end_data}
    ).drop_duplicates("nr")

    nlog_cores = nlog_cores.merge(surface_end_df, on="nr", how="left")
    nlog_cores = adjust_z_coordinates(nlog_cores)

    collection = LayeredData(nlog_cores, has_inclined=True).to_collection(28992, 5709)

    return collection


def read_xml_geotechnical_cores(
    file_or_folder: str | WindowsPath,
) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read xml files of BRO geotechnical boreholes (IMBRO or IMBRO/A quality).
    Decribed in NEN14688 standards

    """
    pass


def read_xml_soil_cores(file_or_folder: str | WindowsPath) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read xml files of BRO soil boreholes (IMBRO or IMBRO/A quality).

    """
    pass


def read_xml_geological_cores(file_or_folder: str | WindowsPath) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read xml files of DINO geological boreholes.

    """
    pass


def read_gef_cores(file_or_folder: str | WindowsPath) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read gef files of boreholes.

    """
    pass


def read_gef_cpts(file_or_folder: str | WindowsPath, use_pygef=False) -> CptCollection:
    """
    Read gef files of CPT data into a geost CptCollection.

    Parameters
    ----------
    file_or_folder : str | WindowsPath
        DESCRIPTION.
    use_pygef : Boolean, optional
        If True, the gef reader from pygef (external) is used. If False, the geost
        gef reader is used. The default is False.

    Returns
    -------
    CptCollection
        DESCRIPTION.

    """
    data = _parse_cpt_gef_files(file_or_folder)
    df = pd.concat(data)

    return CptCollection(df)


def read_xml_cpts(file_or_folder: str | WindowsPath) -> CptCollection:
    """
    NOTIMPLEMENTED
    Read xml files of cpts.

    """
    pass


def get_bro_objects_from_bbox(
    object_type: str,
    xmin: int | float,
    xmax: int | float,
    ymin: int | float,
    ymax: int | float,
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> BoreholeCollection:
    """
    Directly download objects from the BRO to a geost collection object based on the
    given bounding box

    Parameters
    ----------
    object_type : str
        BRO object type to retrieve: BHR-GT, BHR-P, BHR-G or BHR-CPT
    xmin : int | float
        minimal x-coordinate of the bounding box
    xmax : int | float
        maximal x-coordinate of the bounding box
    ymin : int | float
        minimal y-coordinate of the bounding box
    ymax : int | float
        maximal y-coordinate of the bounding box
    vertical_reference : str, optional
        Vertical reference system to use for the resulting collection, see
        :py:attr:`~geost.base.PointDataCollection.vertical_reference`, by default "NAP"
    horizontal_reference : int, optional
        Horizontal reference system to use for the resulting collection, see
        :py:attr:`~geost.base.PointDataCollection.horizontal_reference`,
        by default 28992

    Returns
    -------
    BoreholeCollection
        Instance of either :class:`~geost.borehole.BoreholeCollection` or
        :class:`~geost.borehole.CptCollection` containing only objects selected by
        this method.
    """
    api = BroApi()
    api.search_objects_in_bbox(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        object_type=object_type,
    )
    bro_objects = api.get_objects(api.object_list, object_type=object_type)
    bro_parsed_objects = []
    # TODO: The below has to be adjusted for different BRO objects
    for bro_object in bro_objects:
        try:
            object = SoilCore(bro_object)
        except (TypeError, AttributeError) as err:
            print("Cant read a soil core")
            print(err)
        bro_parsed_objects.append(object.df)

    dataframe = pd.concat(bro_parsed_objects).reset_index()

    collection = BoreholeCollection(
        dataframe,
        vertical_reference="depth",
        horizontal_reference=28992,
        is_inclined=False,
    )
    collection.change_vertical_reference(vertical_reference)
    collection.change_horizontal_reference(horizontal_reference)

    return collection


def get_bro_objects_from_geometry(
    object_type: str,
    geometry_file: Path | str,
    buffer: int | float = 0,
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> BoreholeCollection:
    """
    Directly download objects from the BRO to a geost collection object based on given
    geometries such as points, lines or polygons.

    Parameters
    ----------
    object_type : str
        BRO object type to retrieve: BHR-GT, BHR-P, BHR-G or BHR-CPT
    geometry_file : Path | str
        Path to geometry file. i.e a point/line/polygon shapefile/geopackage/geoparquet
    buffer : int | float, optional
        Buffer distance. Required to be larger than 0 for line and point geometries.
        Adds an extra buffer zone around a polygon to select from, by default 0
    vertical_reference : str, optional
        Vertical reference system to use for the resulting collection, see
        :py:attr:`~geost.base.PointDataCollection.vertical_reference`, by default "NAP"
    horizontal_reference : int, optional
        Horizontal reference system to use for the resulting collection, see,
        :py:attr:`~geost.base.PointDataCollection.horizontal_reference`,
        by default 28992

    Returns
    -------
    BoreholeCollection
        Instance of either :class:`~geost.borehole.BoreholeCollection` or
        :class:`~geost.borehole.CptCollection` containing only objects selected by
        this method.
    """
    file = Path(geometry_file)
    if file.suffix == ".parquet":
        geometry = gpd.read_parquet(geometry_file)
    else:
        geometry = gpd.read_file(geometry_file)
    api = BroApi()
    api.search_objects_in_bbox(
        xmin=geometry.bounds.minx.values[0],
        xmax=geometry.bounds.maxx.values[0],
        ymin=geometry.bounds.miny.values[0],
        ymax=geometry.bounds.maxy.values[0],
        object_type=object_type,
    )
    bro_objects = api.get_objects(api.object_list, object_type=object_type)
    bro_parsed_objects = []
    # TODO: The below has to be adjusted for different BRO objects
    for i, bro_object in enumerate(bro_objects):
        try:
            object = SoilCore(bro_object)
            print(i)
        except (TypeError, AttributeError) as err:
            print("Cant read a soil core")
            print(err)
        bro_parsed_objects.append(object.df)

    dataframe = pd.concat(bro_parsed_objects).reset_index()

    collection = BoreholeCollection(
        dataframe,
        vertical_reference="depth",
        horizontal_reference=28992,
        is_inclined=False,
    )
    collection = getattr(collection, geometry_to_selection_function[geometry.type[0]])(
        geometry, buffer
    )
    collection.header = collection.header.reset_index()
    collection.change_vertical_reference(vertical_reference)
    collection.change_horizontal_reference(horizontal_reference)

    return collection
