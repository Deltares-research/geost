import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path, WindowsPath
from pysst.borehole import BoreholeCollection
from pysst.readers import BroBoreholeReaders, CptXmlReaders
from pysst.utils import get_path_iterable
from pysst.validate import BoreholeSchema, EntriesdataSchema
from typing import Union
from pygef import Cpt

from time import perf_counter


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


def __get_sst_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entries from a table of borehole layers

    Returns
    -------
    pd.DataFrame
        Dataframe containing entries (one line with metadata per borehole/CPT)
    """
    entries = pd.DataFrame(
        [
            [ind.nr, ind.x, ind.y, ind.mv, ind.end]
            for ind in df[["nr", "x", "y", "mv", "end", "top"]].itertuples()
            if ind.top == 0 or ind.mv == ind.top
        ],
        columns=[["nr", "x", "y", "mv", "end"]],
    )
    return entries


def read_sst_cores(file: Union[str, WindowsPath]) -> BoreholeCollection:
    """
    Read Subsurface Toolbox native parquet file with core information

    Parameters
    ----------
    file : Union[str, WindowsPath]
        Path to file to be read.

    Returns
    -------
    BoreholeCollection
        Instance of BoreholeCollection.
    """
    sst_cores = __read_parquet(Path(file))
    BoreholeSchema.validate(sst_cores, inplace=True)
    entries = __get_sst_entries(sst_cores)
    EntriesdataSchema.validate(entries, inplace=True)
    return BoreholeCollection(sst_cores, entries)


def read_sst_cpts(file: Union[str, WindowsPath]):
    """
    Read Subsurface Toolbox native parquet file with cpt information.
    """
    filepath = Path(file)
    sst_cpts = __read_parquet(filepath)


def read_xml_geotechnical(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of BRO geotechnical boreholes (IMBRO or IMBRO/A quality).
    """
    BroBoreholeReaders.xsboringen
    pass


def read_xml_soil(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of BRO soil boreholes (IMBRO or IMBRO/A quality).
    """
    pass


def read_xml_geological(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of DINO geological boreholes.
    """
    pass


def read_gef_cpt(file_or_folder: Union[str, WindowsPath]):
    """
    Read gef files of cpts.
    """
    for gef_file in get_path_iterable(Path(file_or_folder), wildcard="*.gef"):
        gef_cpt = Cpt(str(gef_file))
    pass


def read_xml_cpt(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of cpts.
    """
    pass
