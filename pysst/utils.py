import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path, WindowsPath
from typing import Union

from collections import defaultdict


def csv_to_parquet(
    file: Union[str, WindowsPath], out_file: Union[str, WindowsPath] = None, **kwargs
) -> None:
    """
    Convert csv table to parquet

    Takes pandas.read_csv keyword arguments
    """
    file = Path(file)
    if file.suffix != ".csv":
        raise TypeError(f"File must be a csv file, but a {file.suffix}-file was given")

    df = pd.read_csv(file, **kwargs)
    if out_file is None:
        df.to_parquet(file.parent / (file.stem + ".parquet"))
    else:
        df.to_parquet(out_file)


def excel_to_parquet(
    file: Union[str, WindowsPath], out_file: Union[str, WindowsPath] = None, **kwargs
) -> None:
    """
    Convert Excel table to parquet

    Takes pandas.read_excel keyword arguments
    """
    file = Path(file)
    if not file.suffix in [".xlsx", ".xls"]:
        raise TypeError(
            f"File must be an excel file, but a {file.suffix}-file was given"
        )

    df = pd.read_excel(file, **kwargs)
    if out_file is None:
        df.to_parquet(file.parent / (file.stem + ".parquet"))
    else:
        df.to_parquet(out_file)
