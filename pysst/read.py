from pathlib import Path, WindowsPath
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd

# Local imports
from pysst.borehole import BoreholeCollection, CptCollection
from pysst.bro import BroApi
from pysst.io import _parse_cpt_gef_files
from pysst.io.parsers import SoilCore
from pysst.spatial import header_to_geopandas


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


def read_sst_cores(
    file: str | WindowsPath,
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> BoreholeCollection:
    """
    Read Subsurface Toolbox native parquet file with core information..

    Parameters
    ----------
    file : str | WindowsPath
        Path to file to be read.
    vertical_reference: str
        Which vertical reference is used for tops and bottoms. See
        :py:attr:`~pysst.base.PointDataCollection.vertical_reference` for documentation
        of this attribute.
    horizontal_reference (int): Horizontal reference, see
        :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`

    Returns
    -------
    :class:`~pysst.borehole.BoreholeCollection`
        Instance of :class:`~pysst.borehole.BoreholeCollection`.
    """
    sst_cores = __read_parquet(Path(file))
    return BoreholeCollection(
        sst_cores,
        vertical_reference=vertical_reference,
        horizontal_reference=horizontal_reference,
    )


def read_sst_cpts(
    file: str | WindowsPath,
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> CptCollection:
    """
    Read Subsurface Toolbox native parquet file with cpt information.

    Parameters
    ----------
    file : str | WindowsPath
        Path to file to be read.
    vertical_reference: str
        Which vertical reference is used for tops and bottoms. See
        :py:attr:`~pysst.base.PointDataCollection.vertical_reference` for documentation
        of this attribute.
    horizontal_reference (int): Horizontal reference, see
        :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`

    Returns
    -------
    :class:`~pysst.borehole.CptCollection`
        Instance of :class:`~pysst.borehole.CptCollection`.
    """
    filepath = Path(file)
    sst_cpts = __read_parquet(filepath)
    return CptCollection(
        sst_cpts,
        vertical_reference=vertical_reference,
        horizontal_reference=horizontal_reference,
    )


def read_nlog_cores(
    file: str | WindowsPath, horizontal_reference: int = 28992
) -> BoreholeCollection:
    """
    Read NLog boreholes from the 'nlog_stratstelsel' Excel file. You can find this
    distribution of borehole data here: https://www.nlog.nl/boringen

    Warning: reading this Excel file is really slow (~10 minutes). Converting it to
    parquet using :func:`~pysst.utils.excel_to_parquet` and using that file instead
    allows for much faster reading of nlog data.

    Parameters
    ----------
    file : str | WindowsPath
        Path to nlog_stratstelsel.xlsx or .parquet
    horizontal_reference (int): Horizontal reference, see
        :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`

    Returns
    -------
    :class:`~pysst.borehole.BoreholeCollection`
        :class:`~pysst.borehole.BoreholeCollection`
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
    mv = []
    end = []
    nrs_data = []
    mv_data = []
    end_data = []
    for nr, data in nlog_cores.groupby("nr"):
        nrs.append(nr)
        x.append(data["x"].iloc[0])
        y.append(data["y"].iloc[0])
        x_bot.append(data["x_bot"].iloc[0])
        y_bot.append(data["y_bot"].iloc[0])
        mv.append(data["top"].iloc[0])
        end.append(data["bottom"].iloc[-1])
        nrs_data += [nr for i in range(len(data))]
        mv_data += [data["top"].iloc[0] for i in range(len(data))]
        end_data += [data["bottom"].iloc[-1] for i in range(len(data))]

    mv_end_df = pd.DataFrame(
        {"nr": nrs_data, "mv": mv_data, "end": end_data}
    ).drop_duplicates("nr")

    nlog_cores = nlog_cores.merge(mv_end_df, on="nr", how="left")

    header = header_to_geopandas(
        pd.DataFrame({"nr": nrs, "x": x, "y": y, "mv": mv, "end": end}),
        horizontal_reference,
    )

    return BoreholeCollection(
        nlog_cores,
        vertical_reference="NAP",
        horizontal_reference=horizontal_reference,
        header=header,
        is_inclined=True,
    )


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
    Read gef files of CPT data into a Pysst CptCollection.

    Parameters
    ----------
    file_or_folder : str | WindowsPath
        DESCRIPTION.
    use_pygef : Boolean, optional
        If True, the gef reader from pygef (external) is used. If False, the pysst
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
    Directly download objects from the BRO to a PySST collection object based on the
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
        :py:attr:`~pysst.base.PointDataCollection.vertical_reference`, by default "NAP"
    horizontal_reference : int, optional
        Horizontal reference system to use for the resulting collection, see
        :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`,
        by default 28992

    Returns
    -------
    BoreholeCollection
        Instance of either :class:`~pysst.borehole.BoreholeCollection` or
        :class:`~pysst.borehole.CptCollection` containing only objects selected by
        this method.
    """
    api = BroApi()
    bro_ids = api.search_objects_in_bbox(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        object_type=object_type,
    )
    bro_objects = api.get_objects(bro_ids, object_type=object_type)
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
        horizontal_reference=horizontal_reference,
        is_inclined=False,
    )
    collection.change_vertical_reference(vertical_reference)

    return collection


def get_bro_objects_from_geometry(
    object_type: str,
    geometry_file: Path | str,
    buffer: int | float = 0,
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> BoreholeCollection:
    """
    Directly download objects from the BRO to a PySST collection object based on given
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
        :py:attr:`~pysst.base.PointDataCollection.vertical_reference`, by default "NAP"
    horizontal_reference : int, optional
        Horizontal reference system to use for the resulting collection, see,
        :py:attr:`~pysst.base.PointDataCollection.horizontal_reference`,
        by default 28992

    Returns
    -------
    BoreholeCollection
        Instance of either :class:`~pysst.borehole.BoreholeCollection` or
        :class:`~pysst.borehole.CptCollection` containing only objects selected by
        this method.
    """
    file = Path(geometry_file)
    if file.suffix == ".parquet":
        geometry = gpd.read_parquet(geometry_file)
    else:
        geometry = gpd.read_file(geometry_file)
    api = BroApi()
    bro_ids = api.search_objects_in_bbox(
        xmin=geometry.bounds.minx.values[0],
        xmax=geometry.bounds.maxx.values[0],
        ymin=geometry.bounds.miny.values[0],
        ymax=geometry.bounds.maxy.values[0],
        object_type=object_type,
    )
    bro_objects = api.get_objects(bro_ids, object_type=object_type)
    bro_parsed_objects = []
    # TODO: The below has to be adjusted for different BRO objects
    for bro_object in bro_objects:
        try:
            object = SoilCore(bro_object)
        except (TypeError, AttributeError):  # as err:
            pass
            # print("Cant read a soil core")  # supressed for demo
            # print(err)
        bro_parsed_objects.append(object.df)

    dataframe = pd.concat(bro_parsed_objects).reset_index()

    collection = BoreholeCollection(
        dataframe,
        vertical_reference="depth",
        horizontal_reference=horizontal_reference,
        is_inclined=False,
    )
    collection = getattr(collection, geometry_to_selection_function[geometry.type[0]])(
        geometry, buffer
    )
    collection.header = collection.header.reset_index()
    collection.change_vertical_reference(vertical_reference)

    return collection
