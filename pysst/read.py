from pathlib import Path, WindowsPath
from typing import Union

import numpy as np
import pandas as pd

# Local imports
from pysst.borehole import BoreholeCollection, CptCollection
from pysst.readers import pygef_gef_cpt
from pysst.spatial import header_to_geopandas


def __read_parquet(file: WindowsPath) -> pd.DataFrame:
    """Read parquet file

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
    file: Union[str, WindowsPath],
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> BoreholeCollection:
    """
    Read Subsurface Toolbox native parquet file with core information.

    Parameters
    ----------
    file : Union[str, WindowsPath]
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
    file: Union[str, WindowsPath],
    vertical_reference: str = "NAP",
    horizontal_reference: int = 28992,
) -> CptCollection:
    """
    Read Subsurface Toolbox native parquet file with cpt information.

    Parameters
    ----------
    file : Union[str, WindowsPath]
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
    file: Union[str, WindowsPath], horizontal_reference: int = 28992
) -> BoreholeCollection:
    """
    Read NLog boreholes from the 'nlog_stratstelsel' Excel file. You can find this
    distribution of borehole data here: https://www.nlog.nl/boringen

    Warning: reading this Excel file is really slow (~10 minutes). Converting it to
    parquet using :func:`~pysst.utils.excel_to_parquet` and using that file instead
    allows for much faster reading of nlog data.

    Parameters
    ----------
    file : Union[str, WindowsPath]
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
    mv = []
    end = []
    nrs_data = []
    mv_data = []
    end_data = []
    for nr, data in nlog_cores.groupby("nr"):
        nrs.append(nr)
        x.append(data["x"].iloc[0])
        y.append(data["y"].iloc[0])
        mv.append(data["top"].iloc[0])
        end.append(data["bottom"].iloc[-1])
        nrs_data += [nr for i in range(len(data))]
        mv_data += [data["top"].iloc[0] for i in range(len(data))]
        end_data += [data["bottom"].iloc[-1] for i in range(len(data))]

    mv_end_df = pd.DataFrame({"nr": nrs_data, "mv": mv_data, "end": end_data})

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
    )


def read_xml_geotechnical_cores(
    file_or_folder: Union[str, WindowsPath]
) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read xml files of BRO geotechnical boreholes (IMBRO or IMBRO/A quality).
    Decribed in NEN14688 standards
    """
    pass


def read_xml_soil_cores(file_or_folder: Union[str, WindowsPath]) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read xml files of BRO soil boreholes (IMBRO or IMBRO/A quality).
    """
    pass


def read_xml_geological_cores(
    file_or_folder: Union[str, WindowsPath]
) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read xml files of DINO geological boreholes.
    """
    pass


def read_gef_cores(file_or_folder: Union[str, WindowsPath]) -> BoreholeCollection:
    """
    NOTIMPLEMENTED
    Read gef files of boreholes.
    """
    pass


def read_gef_cpts(file_or_folder: Union[str, WindowsPath]) -> CptCollection:
    """
    NOTIMPLEMENTED
    Read gef files of cpts.
    """
    return CptCollection(pd.concat(pygef_gef_cpt(Path(file_or_folder))))


def read_xml_cpts(file_or_folder: Union[str, WindowsPath]) -> CptCollection:
    """
    NOTIMPLEMENTED
    Read xml files of cpts.
    """
    pass
