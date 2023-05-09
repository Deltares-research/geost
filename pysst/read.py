from pathlib import Path, WindowsPath
from typing import Union

import pandas as pd

# Local imports
from pysst.borehole import BoreholeCollection, CptCollection
from pysst.readers import pygef_gef_cpt


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
    file: Union[str, WindowsPath], vertical_reference: str = "NAP"
) -> BoreholeCollection:
    """
    Read Subsurface Toolbox native parquet file with core information

    Parameters
    ----------
    file : Union[str, WindowsPath]
        Path to file to be read.
    vertical_reference: str
        Which vertical reference is used for tops and bottoms. Either
        'NAP', 'surfacelevel' or 'depth'.

        NAP = elevation with respect to NAP datum.
        surfacelevel = elevation with respect to surface (surface is 0 m, e.g.
                        layers tops could be 0, -1, -2 etc.).
        depth = depth with respect to surface (surface is 0 m, e.g. depth of layers
                tops could be 0, 1, 2 etc.).

    Returns
    -------
    BoreholeCollection
        Instance of BoreholeCollection.
    """
    sst_cores = __read_parquet(Path(file))
    return BoreholeCollection(sst_cores, vertical_reference=vertical_reference)


def read_sst_cpts(
    file: Union[str, WindowsPath], vertical_reference: str = "NAP"
) -> CptCollection:
    """
    Read Subsurface Toolbox native parquet file with cpt information.
    """
    filepath = Path(file)
    sst_cpts = __read_parquet(filepath)
    return CptCollection(sst_cpts, vertical_reference=vertical_reference)


def read_xml_geotechnical_cores(
    file_or_folder: Union[str, WindowsPath]
) -> BoreholeCollection:
    """
    Read xml files of BRO geotechnical boreholes (IMBRO or IMBRO/A quality).
    Decribed in NEN14688 standards
    """
    pass


def read_xml_soil_cores(file_or_folder: Union[str, WindowsPath]) -> BoreholeCollection:
    """
    Read xml files of BRO soil boreholes (IMBRO or IMBRO/A quality).
    """
    pass


def read_xml_geological_cores(
    file_or_folder: Union[str, WindowsPath]
) -> BoreholeCollection:
    """
    Read xml files of DINO geological boreholes.
    """
    pass


def read_gef_cores(file_or_folder: Union[str, WindowsPath]) -> BoreholeCollection:
    """
    Read gef files of boreholes.
    """
    pass


def read_gef_cpts(file_or_folder: Union[str, WindowsPath]) -> CptCollection:
    """
    Read gef files of cpts.
    """
    return CptCollection(pd.concat(pygef_gef_cpt(Path(file_or_folder))))


def read_xml_cpts(file_or_folder: Union[str, WindowsPath]) -> CptCollection:
    """
    Read xml files of cpts.
    """
    pass
