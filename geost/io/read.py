import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from geost import utils
from geost.base import (
    BoreholeCollection,
    Collection,
    CptCollection,
    DiscreteData,
    LayeredData,
    PointHeader,
)
from geost.bro import BroApi
from geost.io import _parse_cpt_gef_files, xml
from geost.io.parsers import BorisXML

type BroObject = Literal["BHR-GT", "BHR-P", "BHR-G", "CPT", "SFR"]

MANDATORY_LAYERED_DATA_COLUMNS = ["nr", "x", "y", "surface", "end", "top", "bottom"]
MANDATORY_DISCRETE_DATA_COLUMNS = ["nr", "x", "y", "surface", "end", "depth"]
MANDATORY_INCLINED_LAYERED_DATA_COLUMNS = [
    "nr",
    "x",
    "y",
    "x_bot",
    "y_bot",
    "surface",
    "end",
    "top",
    "bottom",
]
GEOMETRY_TO_SELECTION_FUNCTION = {
    "Polygon": "select_within_polygons",
    "Line": "select_with_lines",
    "Point": "select_with_points",
}
LLG_COLUMN_MAPPING = {
    "BOORP": "nr",
    "XCO": "x",
    "YCO": "y",
    "MV_HoogteNAP": "surface",
    "BoringEinddiepteNAP": "end",
    "BEGIN_DIEPTE": "top",
    "EIND_DIEPTE": "bottom",
}


def adjust_z_coordinates(data_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Interpret data containing elevation or z-coordinates. GeoST data objects require
    that layer top/bottom or discrete data point z-coordinates to be increasing
    downward, starting at at 0. This function detects how elevation is currently defined
    and turns it into the desired format if required.

    Parameters
    ----------
    data_dataframe : pd.DataFrame
        Dataframe with layer or discrete data that includes columns referecing layer
        top, bottoms or discrete data z-coordinates.
    """
    top_column_label = [
        col for col in data_dataframe.columns if col in ["top", "depth"]
    ][0]
    has_bottom_column = "bottom" in data_dataframe.columns

    # TODO: think about detection. Only considers first two indices to determine
    # downward decreasing or increasing of z-coordinates.
    first_surface = data_dataframe["surface"].iloc[0]
    first_top = data_dataframe[top_column_label].iloc[0]
    second_top = data_dataframe[top_column_label].iloc[1]

    if first_top == first_surface:
        data_dataframe[top_column_label] -= data_dataframe["surface"]
        if has_bottom_column:
            data_dataframe["bottom"] -= data_dataframe["surface"]
    if first_top > second_top:
        data_dataframe[top_column_label] *= -1
        if has_bottom_column:
            data_dataframe["bottom"] *= -1

    return data_dataframe


def _check_mandatory_column_presence(
    df: pd.DataFrame, mandatory_cols: list, column_mapper: dict = None
) -> pd.DataFrame:
    missing_mandatory = [col for col in mandatory_cols if col not in df.columns]

    if missing_mandatory:
        if not column_mapper:
            column_mapper = dict()
            print(
                f"\nMandatory columns {missing_mandatory} are missing in the "
                f"data table. Columns present are: \n{list(df.columns)}\n"
            )
            for missing in missing_mandatory:
                name = input(f"Which column is {missing}? ")
                column_mapper[name] = missing

        df.rename(columns=column_mapper, inplace=True)
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    return df


def read_borehole_table(
    file: str | Path,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
    has_inclined: bool = False,
    as_collection: bool = True,
    column_mapper: dict = None,
    **kwargs,
) -> BoreholeCollection:
    """
    Read tabular borehole information. This is a file (parquet, csv or Excel) that
    includes row data for each (borehole) layer and must at least include the following
    column headers:

    - **nr** : Object id
    - **x** : X-coordinates according to the given horizontal_reference
    - **y** : Y-coordinates according to the given horizontal_reference
    - **surface** : Surface elevation according to the given vertical_reference
    - **end** : End depth according to the given vertical_reference
    - **top** : Top depth of each layer
    - **bottom** : Bottom depth of each layer

    In case there are inclined boreholes in the layer data, you additionally require:

    - **x_bot** : X-coordinates of the layer bottoms
    - **y_bot** : Y-coordinates of the layer bottoms

    If you are reading a file that uses different column names, see the optional
    argument "column_mapper" below. There are no further limits or requirements to
    additional (data) columns.

    Parameters
    ----------
    file : str | Path
        Path to file to be read. Depending on the file extension, the corresponding
        Pandas read function will be called. This can be either pandas.read_parquet,
        pandas.read_csv or pandas.read_excel. The **kwargs that can be given to this
        function will be passed to the selected Panda's read function.
    horizontal_reference : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_reference : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.
    has_inclined : bool, optional
        If True, the borehole data table contains inclined boreholes. The default is False.
    as_collection : bool, optional
        If True, the borehole table will be read as a :class:`~geost.base.Collection`
        which includes a header object and spatial selection functionality. If False,
        a :class:`~geost.base.LayeredData` object is returned. The default is True.
    column_mapper: dict, optional
        If the file to be read uses different column names than the ones given in the
        description above, you can use a dictionary mapping to translate the column
        names to the required format. column_mapper is None by default, in which case
        the user is asked for input if not all required columns are encountered in the
        file.
    kwargs: optional
        Optional keyword arguments for Pandas.read_parquet, Pandas.read_csv or
        Pandas.read_excel.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection` or :class:`~geost.base.LayeredData`
        Instance of :class:`~geost.base.BoreholeCollection` or :class:`~geost.base.LayeredData`
        depending on if the table is read as a collection or not.

    Examples
    --------
    Suppose we have a file of boreholes with columns 'nr', 'x', 'y', 'maaiveld', 'end',
    'top' and 'bottom'. The column 'maaiveld' corresponds to the required column
    'surface' in GeoST and thus needs to be remapped. The x and y-coordinates are given
    in WGS84 UTM 31N whereas the vertical datum is given in the Belgian Ostend height
    vertical datum:

    >>> read_borehole_table(file, 32631, 'Ostend height', column_mapper={'maaiveld': 'surface'})

    """
    boreholes = utils._pandas_read(file, **kwargs)

    if has_inclined:
        boreholes = _check_mandatory_column_presence(
            boreholes, MANDATORY_INCLINED_LAYERED_DATA_COLUMNS, column_mapper
        )
    else:
        boreholes = _check_mandatory_column_presence(
            boreholes, MANDATORY_LAYERED_DATA_COLUMNS, column_mapper
        )
    # Figure out top and bottom reference and coerce to downward increasing from 0 if
    # required.
    boreholes = adjust_z_coordinates(boreholes)
    boreholes = LayeredData(boreholes, has_inclined=has_inclined)

    if as_collection:
        boreholes = boreholes.to_collection(horizontal_reference, vertical_reference)

    return boreholes


def read_cpt_table(
    file: str | Path,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
    as_collection: bool = True,
    column_mapper: dict = None,
    **kwargs,
) -> CptCollection:
    """
    Read tabular CPT information. This is a file (parquet, csv or Excel) that includes
    row data for each CPT depth interval and must at least include the following header
    information as columns:

    - **nr** : Object id
    - **x** : X-coordinates according to the given horizontal_reference
    - **y** : Y-coordinates according to the given horizontal_reference
    - **surface** : Surface elevation according to the given vertical_reference
    - **end** : End depth according to the given vertical_reference
    - **depth** : Depth in meters below the surface

    Parameters
    ----------
    file : str | Path
        File with the CPT information to read.
    horizontal_reference : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_reference : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.
    as_collection : bool, optional
        If True, the CPT table will be read as a :class:`~geost.base.Collection`
        which includes a header object and spatial selection functionality. If False,
        a :class:`~geost.base.DiscreteData` object is returned. The default is True.
    column_mapper: dict, optional
        If the file to be read uses different column names than the ones given in the
        description above, you can use a dictionary mapping to translate the column
        names to the required format. column_mapper is None by default, in which case
        the user is asked for input if not all required columns are encountered in the
        file.
    kwargs: optional
        Optional keyword arguments for Pandas.read_parquet, Pandas.read_csv or
        Pandas.read_excel.

    Returns
    -------
    :class:`~geost.base.CptCollection`
        Instance of :class:`~geost.base.CptCollection`.

    """
    cpts = utils._pandas_read(file, **kwargs)

    cpts = _check_mandatory_column_presence(
        cpts, MANDATORY_DISCRETE_DATA_COLUMNS, column_mapper
    )
    cpts = DiscreteData(cpts)

    if as_collection:
        cpts = cpts.to_collection(horizontal_reference, vertical_reference)

    return cpts


def read_nlog_cores(file: str | Path) -> BoreholeCollection:
    """
    Read NLog boreholes from the 'nlog_stratstelsel' Excel file. You can find this
    distribution of borehole data here: https://www.nlog.nl/boringen

    Warning: reading this Excel file is really slow (~10 minutes). Converting it to
    parquet using :func:`~geost.utils.excel_to_parquet` once and using that file instead
    allows for much faster reading of nlog data.

    Parameters
    ----------
    file : str | Path
        Path to nlog_stratstelsel.xlsx or .parquet

    Returns
    -------
    :class:`~geost.borehole.BoreholeCollection`
        :class:`~geost.borehole.BoreholeCollection`
    """
    nlog_cores = utils._pandas_read(file)

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


def read_xml_boris(
    file: str | Path,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
    as_collection: bool = True,
) -> BoreholeCollection:
    """
    Read export XML of the BORIS software. BORIS is software developed by TNO to
    describe boreholes in the lab. The exported XML-format bears no resemblance to DINO
    or BRO XML standards

    Parameters
    ----------
    file : str | Path
        File with the borehole information to read.
    horizontal_reference : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_reference : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.
    as_collection : bool, optional
        If True, the CPT table will be read as a :class:`~geost.base.Collection`
        which includes a header object and spatial selection functionality. If False,
        a :class:`~geost.base.LayeredData` object is returned. The default is True.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection` or :class:`~geost.base.LayeredData`
        Instance of :class:`~geost.base.BoreholeCollection` or :class:`~geost.base.LayeredData`
        depending on if the table is read as a collection or not.
    """
    boris_data = BorisXML(file)
    boreholes = LayeredData(boris_data.layer_dataframe, has_inclined=False)

    if as_collection:
        # Think of a better way to translate non-standard BORIS crs encoding or keep it
        # user-defined (like we do in other reader functions)
        boreholes = boreholes.to_collection(horizontal_reference, vertical_reference)

    return boreholes


def read_bhrgt(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> BoreholeCollection:
    """
    Read Geotechnical borehole (BHR-GT) XML file or files into a `BoreholeCollection`. This
    reads the XML files according to a `schema` that describes the structure of the of the
    XML data to find the relevant borehole data. Geost provides predefined schemas for the
    BHR-GT XML files that are used in the Netherlands from the Dutch national BRO database.
    A predefined schema may also be present for a specific company, which can be specified
    using the `company` parameter. Predefined schemas from Geost can be used and extended
    or you can provide your own schema to extract relevant attributes (see example link
    below).

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        File or files to read.
    company : str | None, optional
        Company name to use a predefined schema for. See :mod:`geost.io.xml.schemas` for
        available schemas. If None, the predefined schema for BRO BHR-GT XML files is used
        by default to read the data. The default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO BHR-GT XML files
        is used.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`
        BoreholeCollection containing the header and data attributes of the BHR-GT data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://geost.readthedocs.io/en/latest/examples/bhrgt.html

    """
    header, data = xml.read(
        files, xml.read_bhrgt, company=company, schema=schema, read_all=read_all
    )

    header = PointHeader(
        gpd.GeoDataFrame(
            header, geometry=gpd.points_from_xy(header.x, header.y), crs=28992
        ),
        vertical_reference=5709,
    )
    data = LayeredData(data)

    return BoreholeCollection(header, data)


def read_bhrp(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> BoreholeCollection:
    """
    Read Pedological borehole (BHR-P) XML file or files into a `BoreholeCollection`. This
    reads the XML files according to a `schema` that describes the structure of the of the
    XML data to find the relevant borehole data. Geost provides predefined schemas for the
    BHR-P XML files that are used in the Netherlands from the Dutch national BRO database.
    A predefined schema may also be present for a specific company, which can be specified
    using the `company` parameter. Predefined schemas from Geost can be used and extended
    or you can provide your own schema to extract relevant attributes (see example link
    below).

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        File or files to read.
    company : str | None, optional
        Company name to use a predefined schema for. See :mod:`geost.io.xml.schemas` for
        available schemas. If None, the predefined schema for BRO BHR-P XML files is used
        by default to read the data. The default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO BHR-P XML files
        is used.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`
        BoreholeCollection containing the header and data attributes of the BHR-P data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://geost.readthedocs.io/en/latest/examples/bhrp.html

    """
    header, data = xml.read(
        files, xml.read_bhrp, company=company, schema=schema, read_all=read_all
    )
    header = PointHeader(
        gpd.GeoDataFrame(
            header, geometry=gpd.points_from_xy(header.x, header.y), crs=28992
        ),
        vertical_reference=5709,
    )
    data = LayeredData(data)

    return BoreholeCollection(header, data)


def read_bhrg(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> BoreholeCollection:
    """
    Read Geological borehole (BHR-G) XML file or files into a `BoreholeCollection`. This
    reads the XML files according to a `schema` that describes the structure of the of the
    XML data to find the relevant borehole data. Geost provides predefined schemas for the
    BHR-G) XML file or files into a BoreholeCollection. This XML files that are used in the
    Netherlands from the Dutch national BRO database. A predefined schema may also be present
    for a specific company, which can be specified using the `company` parameter. Predefined
    schemas from Geost can be used and extended or you can provide your own schema to extract
    relevant attributes (see example link below).

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        File or files to read.
    company : str | None, optional
        Company name to use a predefined schema for. See :mod:`geost.io.xml.schemas` for
        available schemas. If None, the predefined schema for BRO BHR-G) XML file or files
        into a BoreholeCollection. This XML files is used by default to read the data. The
        default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO BHR-G) XML file
        or files into a BoreholeCollection. This XML files is used.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`
        BoreholeCollection containing the header and data attributes of the BHR-G data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://geost.readthedocs.io/en/latest/examples/bhrgt.html

    """
    header, data = xml.read(
        files, xml.read_bhrg, company=company, schema=schema, read_all=read_all
    )
    header = PointHeader(
        gpd.GeoDataFrame(
            header, geometry=gpd.points_from_xy(header.x, header.y), crs=28992
        ),
        vertical_reference=5709,
    )
    data = LayeredData(data)

    return BoreholeCollection(header, data)


def read_sfr(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> BoreholeCollection:
    """
    Read Pedological soilprofile description (SFR) XML file or files into a `BoreholeCollection`.
    This reads the XML files according to a `schema` that describes the structure of the of
    the XML data to find the relevant borehole data. Geost provides predefined schemas for
    the SFR XML file or files into a BoreholeCollection. This XML files that are used in the
    Netherlands from the Dutch national BRO database. A predefined schema may also be present
    for a specific company, which can be specified using the `company` parameter. Predefined
    schemas from Geost can be used and extended or you can provide your own schema to extract
    relevant attributes (see example link below).

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        File or files to read.
    company : str | None, optional
        Company name to use a predefined schema for. See :mod:`geost.io.xml.schemas` for
        available schemas. If None, the predefined schema for BRO SFR XML file or files
        into a BoreholeCollection. This XML files is used by default to read the data. The
        default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO SFR XML file or
        files into a BoreholeCollection. This XML files is used.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`
        BoreholeCollection containing the header and data attributes of the SFR data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://geost.readthedocs.io/en/latest/examples/bhrgt.html

    """
    header, data = xml.read(
        files, xml.read_sfr, company=company, schema=schema, read_all=read_all
    )
    header = PointHeader(
        gpd.GeoDataFrame(
            header, geometry=gpd.points_from_xy(header.x, header.y), crs=28992
        ),
        vertical_reference=5709,
    )
    data = LayeredData(data)

    return BoreholeCollection(header, data)


def read_gef_cpts(file_or_folder: str | Path) -> CptCollection:
    """
    Read one or more GEF files of CPT data into a geost CptCollection.

    Parameters
    ----------
    file_or_folder : str | Path
        GEF files to read.

    Returns
    -------
    :class:`~geost.base.CptCollection`
        :class:`~geost.base.CptCollection` of the GEF file(s).

    """
    data = _parse_cpt_gef_files(file_or_folder)
    df = pd.concat(data)

    return DiscreteData(df).to_collection()


def read_cpt(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    read_all: bool = False,
) -> CptCollection:
    """
    Read Cone Penetration Test (CPT) XML file or files into a `CptCollection`. This reads
    the XML files according to a `schema` that describes the structure of the of the XML
    data to find the relevant CPT data. Geost provides predefined schemas for the CPT XML
    file or files into a CptCollection. This XML files that are used in the Netherlands
    from the Dutch national BRO database. A predefined schema may also be present for a
    specific company, which can be specified using the `company` parameter. Predefined
    schemas from Geost can be used and extended or you can provide your own schema to
    extract relevant attributes (see example link below).

    Parameters
    ----------
    files : str | Path | Iterable[str  |  Path]
        File or files to read.
    company : str | None, optional
        Company name to use a predefined schema for. See :mod:`geost.io.xml.schemas` for
        available schemas. If None, the predefined schema for BRO CPT XML file or files
        into a CptCollection. This XML files is used by default to read the data. The
        default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO CPT XML file or
        files into a CptCollection. This XML files is used.
    read_all : bool, optional
        XML files may contain multiple CPT elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first CPT element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.CptCollection`
        CptCollection containing the header and data attributes of the CPT data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://geost.readthedocs.io/en/latest/examples/bhrgt.html

    """
    header, data = xml.read(
        files, xml.read_cpt, company=company, schema=schema, read_all=read_all
    )
    header = PointHeader(
        gpd.GeoDataFrame(
            header, geometry=gpd.points_from_xy(header.x, header.y), crs=28992
        ),
        vertical_reference=5709,
    )
    data.fillna({"depth": data["penetrationlength"]}, inplace=True)
    data = DiscreteData(data)

    return CptCollection(header, data)


def read_uullg_tables(
    header_table: str | Path,
    data_table: str | Path,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
    **kwargs,
) -> BoreholeCollection:
    """
    Read the header and data tables from UU-LLG boreholes (see: EasyDans KNAW) from a
    local csv or parquet file into a GeoST BoreholeCollection object.

    Reference dataset: https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:74935
    See https://wikiwfs.geo.uu.nl/ for additional information.

    Parameters
    ----------
    header_table : str | Path
        Path to file of the UU-LLG header table.
    data_table : str | Path
        Path to file of the UU-LLG data table.
    horizontal_reference : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_reference : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.

    Returns
    -------
    :class:`~geost.base.BoreholeCollection`
        Instance of :class:`~geost.base.BoreholeCollection` of the UU-LLG data.

    """
    header = utils._pandas_read(header_table, **kwargs)
    data = utils._pandas_read(data_table, **kwargs)

    header.rename(columns=LLG_COLUMN_MAPPING, inplace=True)
    data.rename(columns=LLG_COLUMN_MAPPING, inplace=True)

    header = header.astype({"nr": str})
    data = data.astype({"nr": str})

    header = utils.dataframe_to_geodataframe(header).set_crs(horizontal_reference)

    add_header_cols = [c for c in ["x", "y", "surface", "end"] if c not in data.columns]
    if add_header_cols:
        data = data.merge(header[["nr"] + add_header_cols], on="nr", how="left")

    header = PointHeader(header, vertical_reference)
    data = LayeredData(data)
    return BoreholeCollection(header, data)


def read_collection_geopackage(
    filepath: str | Path,
    collection_type: Collection,
    horizontal_reference: str | int | CRS = 28992,
    vertical_reference: str | int | CRS = 5709,
):
    """
    Read a GeoST stored Geopackage file of `geost.base.Collection` objects. The
    Geopackage contains the Collection's "header" and "data" attributes as layers which
    are read into the specified Collection.

    Parameters
    ----------
    filepath : str | Path
        GeoST stored Geopackage file of a `geost.base.Collection` object.
    collection_type : Collection
        Type of GeoST Collection object the data needs to be. Subclasses of `Collection`
        (e.g. :class:`geost.base.BoreholeCollection`, :class:`geost.base.CptCollection`)
        for the available types.
    horizontal_reference : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_reference : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.

    Returns
    -------
    Subclass of `geost.base.Collection`
        Subclass which is specified as collection_type.

    Raises
    ------
    ValueError
        Raises ValueError if the collection_type is not supported.

    Examples
    --------
    Any GeoST `Collection` object can be stored as a geopackage file:

    >>> original_collection = geost.data.boreholes_usp()
    >>> original_collection
    BoreholeCollection:
    # header = 67
    >>> original_collection.to_geopackage("./collection.gpkg")

    Reading the stored collection using `geost.read_collection_geopackage` returns the
    stored collection:

    >>> collection = geost.read_collection_geopackage(
    ...     "./collection.gpkg", collection_type=BoreholeCollection
    ...     )
    >>> collection
    BoreholeCollection:
    # header = 67

    """
    # TODO: Check collection type inference possibility from geopackage metadata.
    if collection_type == BoreholeCollection:
        header_type = PointHeader
        data_type = LayeredData
    elif collection_type == CptCollection:
        header_type = PointHeader
        data_type = DiscreteData
    else:
        raise ValueError(f"Collection type {collection_type} not supported.")

    header = gpd.read_file(filepath, layer="header")
    header.set_crs(horizontal_reference, inplace=True, allow_override=True)
    data = gpd.read_file(filepath, layer="data")
    return collection_type(header_type(header, vertical_reference), data_type(data))


def read_pickle(filepath: str | Path, **kwargs) -> Any:
    """
    Read pickled GeoST object (or any object) from file.

    Parameters
    ----------
    filepath : str | Path
        Path to pickle file to open.
    **kwargs
        pd.read_pickle kwargs. See relevant Pandas documentation.

    Returns
    -------
    Any
        Returns the Python object that was stored in the pickle.

    Examples
    --------
    Any GeoST :class:`geost.base.Collection` object can be stored as a pickle file:

    >>> original_collection = geost.data.boreholes_usp()
    >>> original_collection
    BoreholeCollection:
    # header = 67
    >>> original_collection.to_pickle("./collection.pkl")

    Reading the stored pickle using `geost.read_pickle` returns the stored collection:

    >>> collection = geost.read_pickle("./collection.pkl")
    >>> collection
    BoreholeCollection:
    # header = 67

    """
    pkl = pd.read_pickle(filepath, **kwargs)
    return pkl


def bro_api_read(
    object_type: BroObject,
    *,
    bro_ids: str | list[str] = None,
    bbox: tuple[float, float, float, float] = None,
    geometry: gpd.GeoDataFrame = None,
    buffer: int | float = None,
    schema: dict[str, Any] = None,
):
    """
    Read data directly from the BRO API. This allows to read BHR-GT, BHR-P, BHR-G and
    CPT data from the BRO API into `geost` objects without having to download the data
    first. Note, API requests are relatively slow, so this function is not meant for bulk
    downloads. Also, the API is limited to 2000 objects per request, so if you request
    more than 2000 objects, the API will return an error.

    Parameters
    ----------
    object_type : BroObject
        Type of BRO object to read. This can be one of the following: "BHR-GT", "BHR-P",
        "BHR-G", "CPT", "SFR".
    bro_ids : str | list[str], optional (keyword only)
        List of BRO object ids to read. If not given, the bbox or geometry must be
        provided to select the objects to read. If given, bbox and geometry are ignored.
    bbox : tuple[float, float, float, float], optional
        Bounding box (xmin, ymin, xmax, ymax) to search BRO objects within. The default is
        None.
    geometry : gpd.GeoDataFrame, optional (keyword only)
        Geometry to search within. Can be any Shapely geometry object (Point, Line, Polygon),
        Geopandas GeoDataFrame of path to a geometry file. The function retrieve the BRO objects
        that are within the outermost bounds (i.e. bounding box) of the input geometry. The
        default is None.
    buffer : int | float, optional (keyword only)
        Buffer distance to apply to the geometry. This increases the bounding box of the the
        input geometry. The default is None.
    schema : dict[str, Any], optional (keyword only)
        XML schema to use for reading the data which describes the structure of the XML
        data. If not given, the function will use the default schema for the specified
        object_type provided by Geost (see `geost.io.xml.schemas`). Note that predefined
        schemas only retrieve the attributes that are defined in the schema but more data
        may be available in the requested objects. To retrieve desired attributes, you can
        modify the schema accordingly (see usage examples).

    Returns
    -------
    Subclass of `geost.base.Collection`
        Returns a subclass of `geost.base.Collection` depending on the object_type.

    Examples
    --------
    Read a list of specific borehole ids from the BRO API (max 2000 objects):
    >>> import geost
    ... bhrgt = geost.bro_api_read("BHR-GT", bro_ids=["BHR000000339725", "BHR000000339735"])
    ... bhrgt
    BoreholeCollection:
    # header = 2

    Or read geological boreholes (BHR-G) within a bounding box:
    >>> import geost
    ... bbox = (126_800, 448_000, 127_800, 449_000)  # xmin, ymin, xmax, ymax
    ... bhrg = geost.bro_api_read("BHR-G", bbox=bbox)
    BoreholeCollection:
    # header = 3

    Or within shapefile containing a Polygon geometry of the same bounding box of the
    previous example (Point or Line geometries are also supported):
    >>> import geost
    ... bhrg = geost.bro_api_read("BHR-G", geometry="my_polygon.shp")
    BoreholeCollection:
    # header = 3

    Or within the shapefile and applying a buffer:
    >>> import geost
    ... bhrg = geost.bro_api_read("BHR-G", geometry="my_polygon.shp", buffer=500)
    BoreholeCollection:
    # header = 5

    When there are no objects found, you will get an empty collection:
    >>> import geost
    ... bhrg = geost.bro_api_read("BHR-G", bbox=[0, 0, 1, 1])
    Collection:
    <EMPTY COLLECTION>
    """
    readers = {
        "BHR-GT": read_bhrgt,
        "BHR-P": read_bhrp,
        "BHR-G": read_bhrg,
        "CPT": read_cpt,
        "SFR": read_sfr,
    }
    reader = readers.get(object_type, None)
    if reader is None:
        raise ValueError(f"Object type '{object_type}' is not supported for reading.")

    api = BroApi()

    if bro_ids is not None:
        bro_data = api.get_objects(bro_ids, object_type=object_type)
    else:
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
        elif geometry is not None:
            geometry = utils.check_geometry_instance(geometry)
            if buffer is not None:
                geometry = geometry.buffer(buffer)
            xmin, ymin, xmax, ymax = geometry.total_bounds
        api.search_objects_in_bbox(xmin, ymin, xmax, ymax, object_type=object_type)
        bro_data = api.get_objects(api.object_list, object_type=object_type)

    if bro_data:
        collection = reader(bro_data, schema=schema)
    else:
        collection = Collection(None, None)

    return collection
