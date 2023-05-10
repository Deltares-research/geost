# import numpy as np
# import pandas as pd
from typing import Iterable, Union
from pathlib import Path, WindowsPath
from pysst.utils import get_path_iterable
from pysst.io.parsers.gef_parsers import CptGefFile


def _parse_cpt_gef_files(file_or_folder: Union[str, WindowsPath]):
    """
    Parse gef files from CPT data into Pandas DataFrames.

    Parameters
    ----------
    file_or_folder : Union[str, WindowsPath]
        Gef file or folder with gef files to parse.

    Yields
    ------
    df : pd.DataFrame
        Pandas DataFrame per gef file.

    """
    if isinstance(file_or_folder, (str, WindowsPath)):
        files = get_path_iterable(Path(file_or_folder))

    elif isinstance(file_or_folder, Iterable):
        files = file_or_folder

    for f in files:
        cpt = CptGefFile(f)
        df = cpt.df

        # header_cols = pd.DataFrame(
        #     dict(
        #         nr=np.repeat(cpt.nr, len(df)),
        #         x=np.repeat(cpt.x, len(df)),
        #         y=np.repeat(cpt.y, len(df)),
        #         mv=np.repeat(cpt.z, len(df)),
        #         end=np.repeat(cpt.enddepth, len(df)),
        #         )
        #     )

        df.insert(0, 'nr', cpt.nr)
        df.insert(1, 'x', cpt.x)
        df.insert(2, 'y', cpt.y)
        df.insert(3, 'mv', cpt.z)
        df.insert(4, 'end', cpt.enddepth)
        yield df
