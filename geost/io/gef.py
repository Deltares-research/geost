# import numpy as np
# import pandas as pd
from pathlib import Path
from typing import Iterable, Union

from geost.io.parsers.gef_parsers import CptGefFile
from geost.utils import get_path_iterable


def _parse_cpt_gef_files(file_or_folder: Union[str, Path]):
    """
    Parse gef files from CPT data into Pandas DataFrames.

    Parameters
    ----------
    file_or_folder : Union[str, Path]
        Gef file or folder with gef files to parse.

    Yields
    ------
    df : pd.DataFrame
        Pandas DataFrame per gef file.

    """
    if isinstance(file_or_folder, (str, Path)):
        files = get_path_iterable(Path(file_or_folder), wildcard="*.gef")

    elif isinstance(file_or_folder, Iterable):
        files = file_or_folder

    for f in files:
        cpt = CptGefFile(f)
        df = cpt.df

        df.insert(0, "nr", cpt.nr)
        df.insert(1, "x", cpt.x)
        df.insert(2, "y", cpt.y)
        df.insert(3, "surface", cpt.z)
        df.insert(4, "end", cpt.enddepth)
        yield df
