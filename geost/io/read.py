import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from geost._warnings import future_deprecation_warning
from geost.base import Collection
from geost.bro import BroApi
from geost.io import _parse_cpt_gef_files, xml
from geost.io.parsers import BorisXML
from geost.utils import conversion, io_helpers
from geost.validation.column_names import check_positional_column_presence

type BroObject = Literal["BHR-GT", "BHR-GT-samples", "BHR-P", "BHR-G", "CPT", "SFR"]


LLG_COLUMN_MAPPING = {
    "BOORP": "nr",
    "XCO": "x",
    "YCO": "y",
    "MV_HoogteNAP": "surface",
    "BoringEinddiepteNAP": "end",
    "BEGIN_DIEPTE": "top",
    "EIND_DIEPTE": "bottom",
}


def read_table(
    file: str | Path,
    as_collection: bool = True,
    crs: str | int | CRS = None,
    vertical_datum: str | int | CRS = None,
    column_mapper: dict = None,
    pd_kwargs: dict[str, Any] = None,
    coll_kwargs: dict[str, Any] = None,
) -> Collection | pd.DataFrame:  # pragma: no cover
    pass


@future_deprecation_warning(alternative_func=read_table)
def read_borehole_table(
    file: str | Path,
    as_collection: bool = True,
    crs: str | int | CRS = None,
    vertical_datum: str | int | CRS = None,
    column_mapper: dict = None,
    pd_kwargs: dict[str, Any] = None,
    coll_kwargs: dict[str, Any] = None,
) -> Collection | pd.DataFrame:
    """
    Read tabular borehole information. This is a file (parquet, csv or Excel) that
    includes row data for each (borehole) layer and must at least include the following
    column headers:

    - **nr** : Object id
    - **x** : X-coordinates according to the given crs
    - **y** : Y-coordinates according to the given crs
    - **surface** : Surface elevation according to the given vertical_datum
    - **end** : End depth according to the given vertical_datum
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
        pandas.read_csv or pandas.read_excel. Optional keyword arguments that can be
        given in the specific Pandas read function can be passed via the `pd_kwargs`
        argument.
    as_collection : bool, optional
        If True, the borehole table will be read as a :class:`~geost.base.Collection`.
        Optional keyword arguments that can be given to :meth:`~geost.accessor.GeostFrame.to_collection`
        can be passed via the `coll_kwargs` argument. If False, a `pd.DataFrame` is returned.
        The default is True.
    crs : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is None, which means no CRS will
        be assigned to the resulting Collection. Only used if `as_collection=True`.
    vertical_datum : str | int | CRS, optional
        Vertical datum for the collection. The default is None. Only used if
        `as_collection=True`.
    column_mapper: dict, optional
        If the file to be read uses different column names than the ones given in the
        description above, you can use a dictionary mapping to translate the column
        names to the required format. column_mapper is None by default, in which case
        the user is asked for input if not all required columns are encountered in the
        file.
    pd_kwargs: dict[str, Any], optional
        Optional keyword arguments for Pandas.read_parquet, Pandas.read_csv or
        Pandas.read_excel.
    coll_kwargs: dict[str, Any], optional
        Optional keyword arguments for :meth:`~geost.accessor.GeostFrame.to_collection`

    Returns
    -------
    :class:`~geost.base.Collection` or `pd.DataFrame`
        Instance of :class:`~geost.base.Collection` when `as_collection=True` or `pd.DataFrame`
        otherwise.

    Examples
    --------
    Suppose we have a file of boreholes with columns 'nr', 'x', 'y', 'maaiveld', 'end',
    'top' and 'bottom'. The column 'maaiveld' corresponds to the required column
    'surface' in GeoST and thus needs to be remapped. The x and y-coordinates are given
    in WGS84 UTM 31N whereas the vertical datum is given in the Belgian Ostend height
    vertical datum, so we specify these in the `coll_kwargs` so these will be used in the
    `to_collection` method:

    >>> from geost import read_borehole_table
    >>> collection_kwargs = {
    ...     "include_in_header": ["nr", "x", "y", "surface", "end"],
    ...     "has_inclined": False,
    ... }
    >>> collection = read_borehole_table(
    ...     file, column_mapper={'maaiveld': 'surface'}, coll_kwargs=collection_kwargs
    ... )

    """
    pd_kwargs = pd_kwargs or dict()
    coll_kwargs = coll_kwargs or dict()

    boreholes = io_helpers._pandas_read(file, **pd_kwargs)

    if column_mapper:
        boreholes.rename(columns=column_mapper, inplace=True)

    check_positional_column_presence(boreholes)
    boreholes = conversion.adjust_z_coordinates(boreholes)

    if as_collection:
        boreholes = boreholes.gst.to_collection(
            crs=crs, vertical_datum=vertical_datum, **coll_kwargs
        )

    return boreholes


@future_deprecation_warning(alternative_func=read_table)
def read_cpt_table(
    file: str | Path,
    as_collection: bool = True,
    crs: str | int | CRS = None,
    vertical_datum: str | int | CRS = None,
    column_mapper: dict = None,
    pd_kwargs: dict[str, Any] = None,
    coll_kwargs: dict[str, Any] = None,
) -> Collection | pd.DataFrame:
    """
    Read tabular CPT information. This is a file (parquet, csv or Excel) that includes
    row data for each CPT depth interval and must at least include the following header
    information as columns:

    - **nr** : Object id
    - **x** : X-coordinates according to the given crs
    - **y** : Y-coordinates according to the given crs
    - **surface** : Surface elevation according to the given vertical_datum
    - **end** : End depth according to the given vertical_datum
    - **depth** : Depth in meters below the surface

    Parameters
    ----------
    file : str | Path
        Path to file to be read. Depending on the file extension, the corresponding
        Pandas read function will be called. This can be either pandas.read_parquet,
        pandas.read_csv or pandas.read_excel. Optional keyword arguments that can be
        given in the specific Pandas read function can be passed via the `pd_kwargs`
        argument.
    as_collection : bool, optional
        If True, the borehole table will be read as a :class:`~geost.base.Collection`.
        Optional keyword arguments that can be given to :meth:`~geost.accessor.GeostFrame.to_collection`
        can be passed via the `coll_kwargs` argument. If False, a `pd.DataFrame` is returned.
        The default is True.
    crs : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is None, which means no CRS will
        be assigned to the resulting Collection. Only used if `as_collection=True`.
    vertical_datum : str | int | CRS, optional
        Vertical datum for the collection. The default is None. Only used if
        `as_collection=True`.
    column_mapper: dict, optional
        If the file to be read uses different column names than the ones given in the
        description above, you can use a dictionary mapping to translate the column
        names to the required format. column_mapper is None by default, in which case
        the user is asked for input if not all required columns are encountered in the
        file.
    pd_kwargs: dict[str, Any], optional
        Optional keyword arguments for Pandas.read_parquet, Pandas.read_csv or
        Pandas.read_excel.
    coll_kwargs: dict[str, Any], optional
        Optional keyword arguments for :meth:`~geost.accessor.GeostFrame.to_collection`

    Returns
    -------
    :class:`~geost.base.Collection` or `pd.DataFrame`
        Instance of :class:`~geost.base.Collection` when `as_collection=True` or `pd.DataFrame`
        otherwise.

    """
    pd_kwargs = pd_kwargs or dict()
    coll_kwargs = coll_kwargs or dict()

    cpts = io_helpers._pandas_read(file, **pd_kwargs)

    if column_mapper:
        cpts.rename(columns=column_mapper, inplace=True)

    check_positional_column_presence(cpts)

    if as_collection:
        cpts = cpts.gst.to_collection(
            crs=crs, vertical_datum=vertical_datum, **coll_kwargs
        )

    return cpts


def read_nlog_cores(file: str | Path, **kwargs) -> Collection:
    """
    Read NLog boreholes from the 'nlog_stratstelsel' Excel file. You can find this
    distribution of borehole data here: https://www.nlog.nl/boringen

    Note: reading this Excel file is really slow. Converting it to parquet using
    :func:`~geost.utils.io_helpers.excel_to_parquet` once and using that file instead
    allows for much faster reading of nlog data.

    Parameters
    ----------
    file : str | Path
        Path to nlog_stratstelsel.xlsx or .parquet
    **kwargs
        Optional keyword arguments for Pandas.read_parquet or Pandas.read_excel depending
        on the file extension of `file`.

    Returns
    -------
    :class:`~geost.borehole.Collection`
        :class:`~geost.borehole.Collection`

    """
    data = io_helpers._pandas_read(file, **kwargs)

    data.rename(
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

    data[["top", "bottom"]] *= -1
    header = data.groupby("nr", as_index=False).agg(
        {"x": "first", "y": "first", "top": "max", "bottom": "min"}
    )
    header.rename(columns={"top": "surface", "bottom": "end"}, inplace=True)
    header = gpd.GeoDataFrame(
        header, geometry=gpd.points_from_xy(header["x"], header["y"]), crs=28992
    )

    data = data.merge(header[["nr", "surface", "end"]], on="nr", how="left")
    data = conversion.adjust_z_coordinates(data)

    return Collection(data, header=header, has_inclined=True, vertical_datum=5709)


def read_xml_boris(
    file: str | Path, as_collection: bool = True, **kwargs
) -> Collection:
    """
    Read export XML of the BORIS software. BORIS is software developed by TNO to
    describe boreholes in the lab. The exported XML-format bears no resemblance to DINO
    or BRO XML standards

    Parameters
    ----------
    file : str | Path
        File with the borehole information to read.
    as_collection : bool, optional
        If True, the CPT table will be read as a :class:`~geost.base.Collection`
        which includes a header object and spatial selection functionality. If False,
        a `pandas.DataFrame` object is returned. The default is True.
    **kwargs
        Additional keyword arguments that can be given to :meth:`~geost.accessor.GeostFrame.to_collection`
        when `as_collection=True`.

    Returns
    -------
    :class:`~geost.base.Collection`
        Instance of :class:`~geost.base.Collection` or `pandas.DataFrame`
        depending on if the table is read as a collection or not.

    """
    boreholes = BorisXML(file).layer_dataframe

    if as_collection:
        # Think of a better way to translate non-standard BORIS crs encoding or keep it
        # user-defined (like we do in other reader functions)
        boreholes = boreholes.gst.to_collection(**kwargs)

    return boreholes


def read_bhrgt(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    crs: str | int | CRS = 28992,
    read_all: bool = False,
) -> Collection:
    """
    Read Geotechnical borehole (BHR-GT) XML file or files into a `Collection`. This
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
    crs : str | int | CRS, optional
        EPSG of the desired data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.Collection`
        Collection containing the header and data attributes of the BHR-GT data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://deltares-research.github.io/geost/api_reference/generated/geost.bro_api_read.html

    """
    header, data = xml.read(
        files,
        xml.read_bhrgt,
        company=company,
        schema=schema,
        read_all=read_all,
        coerce_crs=crs,
    )
    header = gpd.GeoDataFrame(
        header,
        geometry=gpd.points_from_xy(header.x, header.y),
        crs=crs,
    )

    return Collection(data, header=header)


def read_bhrgt_samples(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    crs: str | int | CRS = 28992,
    read_all: bool = False,
) -> Collection:
    """
    Read sample data in a Geotechnical borehole (BHR-GT) XML file or files into a
    `Collection`. This reads the XML files according to a `schema` that describes
    the structure of the of the XML data to find the relevant sample data. Geost provides
    predefined schemas for the BHR-GT XML files that are used in the Netherlands from
    the Dutch national BRO database. A predefined schema may also be present for a specific
    company, which can be specified using the `company` parameter. Predefined schemas
    from Geost can be used and extended or you can provide your own schema to extract
    relevant attributes (see example link below).

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
    crs : str | int | CRS, optional
        EPSG of the desired data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.Collection`
        Collection containing the header and data attributes of the BHR-GT data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://deltares-research.github.io/geost/api_reference/generated/bhrgt_samples.html

    """
    header, data = xml.read(
        files,
        xml.read_bhrgt_samples,
        company=company,
        schema=schema,
        read_all=read_all,
        coerce_crs=crs,
    )
    header = gpd.GeoDataFrame(
        header,
        geometry=gpd.points_from_xy(header.x, header.y),
        crs=crs,
    )

    if "beginDepth" in data.columns or "endDepth" in data.columns:
        data.rename(columns={"beginDepth": "top", "endDepth": "bottom"}, inplace=True)
        data[["top", "bottom"]] = data[["top", "bottom"]].astype(float)

    # If no sample data was found, the data table will be empty. In that case, we can
    # just return an empty collection.
    if "top" in data.columns and "bottom" in data.columns:
        return Collection(data, header=header)
    else:
        warnings.warn(
            "No sample data found in the provided XML file(s). Returning an empty collection."
        )
        return Collection(None, header=None)


def read_bhrp(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    crs: str | int | CRS = 28992,
    read_all: bool = False,
) -> Collection:
    """
    Read Pedological borehole (BHR-P) XML file or files into a `Collection`. This
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
    crs: str | int | CRS, optional
        EPSG of the desired data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.Collection`
        Collection containing the header and data attributes of the BHR-P data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://deltares-research.github.io/geost/api_reference/generated/geost.bro_api_read.html

    """
    header, data = xml.read(
        files,
        xml.read_bhrp,
        company=company,
        schema=schema,
        read_all=read_all,
        coerce_crs=crs,
    )
    header = gpd.GeoDataFrame(
        header,
        geometry=gpd.points_from_xy(header.x, header.y),
        crs=crs,
    )

    return Collection(data, header=header)


def read_bhrg(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    crs: str | int | CRS = 28992,
    read_all: bool = False,
) -> Collection:
    """
    Read Geological borehole (BHR-G) XML file or files into a `Collection`. This
    reads the XML files according to a `schema` that describes the structure of the of the
    XML data to find the relevant borehole data. Geost provides predefined schemas for the
    BHR-G) XML file or files into a Collection. This XML files that are used in the
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
        into a Collection. This XML files is used by default to read the data. The
        default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO BHR-G) XML file
        or files into a Collection. This XML files is used.
    crs : str | int | CRS, optional
        EPSG of the desired data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.Collection`
        Collection containing the header and data attributes of the BHR-G data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://deltares-research.github.io/geost/api_reference/generated/geost.bro_api_read.html

    """
    header, data = xml.read(
        files,
        xml.read_bhrg,
        company=company,
        schema=schema,
        read_all=read_all,
        coerce_crs=crs,
    )
    header = gpd.GeoDataFrame(
        header,
        geometry=gpd.points_from_xy(header.x, header.y),
        crs=crs,
    )

    return Collection(data, header=header)


def read_sfr(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    crs: str | int | CRS = 28992,
    read_all: bool = False,
) -> Collection:
    """
    Read Pedological soilprofile description (SFR) XML file or files into a `Collection`.
    This reads the XML files according to a `schema` that describes the structure of the of
    the XML data to find the relevant borehole data. Geost provides predefined schemas for
    the SFR XML file or files into a Collection. This XML files that are used in the
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
        into a Collection. This XML files is used by default to read the data. The
        default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO SFR XML file or
        files into a Collection. This XML files is used.
    crs : str | int | CRS, optional
        EPSG of the desired data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    read_all : bool, optional
        XML files may contain multiple borehole elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first borehole element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.Collection`
        Collection containing the header and data attributes of the SFR data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://deltares-research.github.io/geost/api_reference/generated/geost.bro_api_read.html

    """
    header, data = xml.read(
        files,
        xml.read_sfr,
        company=company,
        schema=schema,
        read_all=read_all,
        coerce_crs=crs,
    )
    header = gpd.GeoDataFrame(
        header,
        geometry=gpd.points_from_xy(header.x, header.y),
        crs=crs,
    )

    return Collection(data, header=header)


def read_gef_cpts(
    file_or_folder: str | Path, as_collection=True, **kwargs
) -> Collection | pd.DataFrame:
    """
    Read one or more GEF files of CPT data into a geost Collection.

    Parameters
    ----------
    file_or_folder : str | Path
        GEF files to read.
    as_collection : bool, optional
        If True, return a :class:`~geost.base.Collection` from the GEF files. If False,
        a `pd.DataFrame` is returned.
    **kwargs
        Additional keyword arguments that can be given to :meth:`~geost.accessor.GeostFrame.to_collection`

    Returns
    -------
    :class:`~geost.base.Collection` or `pd.DataFrame`
        :class:`~geost.base.Collection` of the GEF file(s).

    """
    data = _parse_cpt_gef_files(file_or_folder)
    cpts = pd.concat(data)
    if as_collection:
        cpts = cpts.gst.to_collection(**kwargs)
    return cpts


def read_cpt(
    files: str | Path | Iterable[str | Path],
    company: str | None = None,
    schema: dict[str, Any] = None,
    crs: str | int | CRS = 28992,
    read_all: bool = False,
) -> Collection:
    """
    Read Cone Penetration Test (CPT) XML file or files into a `Collection`. This reads
    the XML files according to a `schema` that describes the structure of the of the XML
    data to find the relevant CPT data. Geost provides predefined schemas for the CPT XML
    file or files into a Collection. This XML files that are used in the Netherlands
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
        into a Collection. This XML files is used by default to read the data. The
        default is None.
    schema : dict[str, Any], optional
        Schema to use for reading the data which describes the structure of the XML data.
        If not given and `company` is None, the predefined schema for BRO CPT XML file or
        files into a Collection. This XML files is used.
    crs: str | int | CRS, optional
        EPSG of the desired data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    read_all : bool, optional
        XML files may contain multiple CPT elements per XML file. If True, the function
        will attempt to read all available data. Otherwise, only the first CPT element
        is read. The default is False.

    Returns
    -------
    :class:`~geost.base.Collection`
        Collection containing the header and data attributes of the CPT data.

    Examples
    --------
    A full example of reading XML files can be found here:
    https://deltares-research.github.io/geost/api_reference/generated/geost.bro_api_read.html

    """
    header, data = xml.read(
        files,
        xml.read_cpt,
        company=company,
        schema=schema,
        read_all=read_all,
        coerce_crs=crs,
    )
    header = gpd.GeoDataFrame(
        header,
        geometry=gpd.points_from_xy(header.x, header.y),
        crs=crs,
    )

    data.fillna({"depth": data["penetrationlength"]}, inplace=True)
    data.sort_values(["nr", "depth"], inplace=True)

    return Collection(data, header=header)


def read_uullg_tables(
    header_table: str | Path,
    data_table: str | Path,
    crs: str | int | CRS = 28992,
    vertical_datum: str | int | CRS = 5709,
    **kwargs,
) -> Collection:
    """
    Read the header and data tables from UU-LLG boreholes (see: EasyDans KNAW) from a
    local csv or parquet file into a GeoST Collection object.

    Reference dataset: https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:74935
    See https://wikiwfs.geo.uu.nl/ for additional information.

    Parameters
    ----------
    header_table : str | Path
        Path to file of the UU-LLG header table.
    data_table : str | Path
        Path to file of the UU-LLG data table.
    crs : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_datum : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.

    Returns
    -------
    :class:`~geost.base.Collection`
        Instance of :class:`~geost.base.Collection` of the UU-LLG data.

    """
    header = io_helpers._pandas_read(header_table, **kwargs)
    data = io_helpers._pandas_read(data_table, **kwargs)

    header.rename(columns=LLG_COLUMN_MAPPING, inplace=True)
    data.rename(columns=LLG_COLUMN_MAPPING, inplace=True)

    header = header.astype({"nr": str})
    data = data.astype({"nr": str})

    header = conversion.dataframe_to_geodataframe(header).set_crs(crs)

    add_header_cols = [c for c in {"x", "y", "surface", "end"} if c not in data.columns]
    if add_header_cols:
        data = data.merge(header[["nr"] + add_header_cols], on="nr", how="left")

    return Collection(data, header=header, vertical_datum=vertical_datum)


def read_collection_geopackage(
    filepath: str | Path,
    has_inclined: bool = False,
    crs: str | int | CRS = 28992,
    vertical_datum: str | int | CRS = 5709,
):
    """
    Read a GeoST stored Geopackage file of `geost.base.Collection` objects. The Geopackage
    contains the Collection's "header" and "data" attributes as layers which are read into
    the specified Collection.

    Parameters
    ----------
    filepath : str | Path
        GeoST stored Geopackage file of a `geost.base.Collection` object.
    has_inclined : bool, optional
        If True, the borehole data table contains inclined data. The default is False.
    crs : str | int | CRS, optional
        EPSG of the data's horizontal reference. Takes anything that can be interpreted
        by pyproj.crs.CRS.from_user_input(). The default is 28992.
    vertical_datum : str | int | CRS, optional
        EPSG of the data's vertical datum. Takes anything that can be interpreted by
        pyproj.crs.CRS.from_user_input(). However, it must be a vertical datum. FYI:
        "NAP" is EPSG 5709 and The Belgian reference system (Ostend height) is ESPG
        5710. The default is 5709.

    Returns
    -------
    `geost.base.Collection`

    Examples
    --------
    Any GeoST `Collection` object can be stored as a geopackage file:

    >>> original_collection = geost.data.boreholes_usp()
    >>> original_collection
    Collection:
    # header = 67
    >>> original_collection.to_geopackage("./collection.gpkg")

    Reading the stored collection using `geost.read_collection_geopackage` returns the
    stored collection:

    >>> collection = geost.read_collection_geopackage("./collection.gpkg")
    >>> collection
    Collection:
    # header = 67

    """
    header = gpd.read_file(filepath, layer="header")
    header.set_crs(crs, inplace=True, allow_override=True)
    data = gpd.read_file(filepath, layer="data")

    return Collection(
        data,
        header=header,
        has_inclined=has_inclined,
        vertical_datum=vertical_datum,
    )


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
    Collection:
    # header = 67
    >>> original_collection.to_pickle("./collection.pkl")

    Reading the stored pickle using `geost.read_pickle` returns the stored collection:

    >>> collection = geost.read_pickle("./collection.pkl")
    >>> collection
    Collection:
    # header = 67

    """
    pkl = pd.read_pickle(filepath, **kwargs)
    return pkl


def bro_api_read(
    object_type: BroObject,
    *,
    bro_ids: str | list[str] | pd.Series = None,
    crs: int | str | CRS = 28992,
    bbox: tuple[float, float, float, float] = None,
    geometry: gpd.GeoDataFrame = None,
    buffer: int | float = None,
    schema: dict[str, Any] = None,
) -> Collection:
    """
    Read data directly from the BRO API. This allows to read BHR-GT, BHR-P, BHR-G and
    CPT data from the BRO API into `geost` objects without having to download the data
    first. Note, API requests are relatively slow, so this function is not meant for bulk
    downloads. Also, the API is limited to 2000 objects per request, so if you request
    more than 2000 objects, the API will return an error.

    Parameters
    ----------
    object_type : BroObject
        Type of BRO object to read. This can be one of the following: "BHR-GT", "BHR-GT-samples",
        "BHR-P", "BHR-G", "CPT", "SFR".
    bro_ids : str | list[str] | pd.Series, optional (keyword only)
        List of BRO object ids to read. If not given, the bbox or geometry must be
        provided to select the objects to read. If given, bbox and geometry are ignored.
    crs : int, optional (keyword only)
        EPSG code of the coordinate reference system that the collection will be coerced to.
        If using a bbox for selection, this crs is also used to interpret the bbox coordinates.
        Note that when using a geometry for selection, the geometry's CRS is used instead unless
        the geometry has no CRS defined. In that case, this crs is also used.
        The default is 28992 (RD New).
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
    :class:`~geost.base.Collection`
        Returns a subclass of `geost.Collection` depending on the object_type.

    Examples
    --------
    Read a list of specific borehole ids from the BRO API (max 2000 objects):

    >>> import geost
    ... bhrgt = geost.bro_api_read("BHR-GT", bro_ids=["BHR000000339725", "BHR000000339735"])
    ... bhrgt
    Collection:
    # header = 2

    Or read geological boreholes (BHR-G) within a bounding box. Note the crs parameter
    is used to interpret the bbox coordinates (by default it is 28992 - RD New):

    >>> import geost
    ... bbox = (126_800, 448_000, 127_800, 449_000)  # xmin, ymin, xmax, ymax
    ... bhrg = geost.bro_api_read("BHR-G", bbox=bbox, crs=28992)
    Collection:
    # header = 3

    Or within shapefile containing a Polygon geometry of the same bounding box of the
    previous example (Point or Line geometries are also supported). Note that the crs
    of the geometry is used if it has a defined CRS, otherwise the crs parameter must be used:

    >>> import geost
    ... bhrg = geost.bro_api_read("BHR-G", geometry="my_polygon.shp")
    Collection:
    # header = 3

    Or within the shapefile and applying a buffer:

    >>> import geost
    ... bhrg = geost.bro_api_read("BHR-G", geometry="my_polygon.shp", buffer=500)
    Collection:
    # header = 5

    When there are no objects found, you will get an empty collection:

    >>> import geost
    ... bhrg = geost.bro_api_read("BHR-G", bbox=[0, 0, 1, 1])
    Collection:
    <EMPTY COLLECTION>

    """
    readers = {
        "BHR-GT": read_bhrgt,
        "BHR-GT-samples": read_bhrgt_samples,
        "BHR-P": read_bhrp,
        "BHR-G": read_bhrg,
        "CPT": read_cpt,
        "SFR": read_sfr,
    }
    if object_type not in readers:
        raise ValueError(f"Object type '{object_type}' is not supported for reading.")

    reader = readers[object_type]

    crs = CRS.from_user_input(crs)

    api = BroApi()

    if bro_ids is not None:
        bro_data = api.get_objects(bro_ids, object_type=object_type)
    else:
        if bbox:
            xmin, ymin, xmax, ymax = bbox
        elif geometry is not None:
            geometry = conversion.check_geometry_instance(geometry)
            crs = CRS.from_user_input(geometry.crs or crs)
            if geometry.crs is None:
                warnings.warn(
                    f"Geometry for retrieving '{object_type}' objects has no CRS defined. "
                    f"Using the provided crs parameter ({crs.to_string()}) to interpret "
                    "the geometry CRS. CHECK IF THIS IS CORRECT!",
                    category=UserWarning,
                )
            if buffer:
                geometry = geometry.buffer(buffer)
            xmin, ymin, xmax, ymax = geometry.total_bounds
        api.search_objects_in_bbox(
            xmin, ymin, xmax, ymax, epsg=str(crs.to_epsg()), object_type=object_type
        )
        bro_data = api.get_objects(api.object_list, object_type=object_type)

    if bro_data:
        collection = reader(bro_data, schema=schema, crs=crs)
    else:
        collection = Collection(None, header=None)

    return collection
