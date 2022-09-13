import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path, WindowsPath
from tqdm import tqdm
from pysst.borehole import BoreholeCollection
from typing import Union


def __read_parquet(file: WindowsPath) -> pd.DataFrame:
    suffix = file.suffix
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file)
    else:
        raise TypeError(
            f"Expected parquet file (with .parquet or .pq suffix) but got {suffix} file"
        )


def read_sst_cores(file: Union[str, WindowsPath]) -> BoreholeCollection:
    """
    Read Subsurface Toolbox native parquet file with core information
    """
    filepath = Path(file)
    sst_cores = __read_parquet(filepath)
    # validate here
    sst_cores.set_index(["nr", "x", "y", "mv", "end", "top"], inplace=True)
    return BoreholeCollection(filepath, sst_cores)


def read_sst_cpts(file: Union[str, WindowsPath]):
    """
    Read Subsurface Toolbox native parquet file with cpt information
    """
    filepath = Path(file)
    sst_cpts = __read_parquet(filepath)


def read_xml_geotechnical(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of BRO geotechnical boreholes (IMBRO or IMBRO/A quality)
    """
    pass


def read_xml_soil(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of BRO soil boreholes (IMBRO or IMBRO/A quality)
    """
    pass


def read_xml_geological(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of DINO geological boreholes
    """
    pass


def read_xml_cpt(file_or_folder: Union[str, WindowsPath]):
    """
    Read xml files of cpts
    """
    pass


def read_gef_cpt(file_or_folder: Union[str, WindowsPath]):
    """
    Read gef files of cpts
    """
    pass
