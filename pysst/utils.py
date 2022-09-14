import time
import numpy as np
import pandas as pd
from pathlib import Path, WindowsPath
from typing import Union

from collections import defaultdict


def csv_to_parquet(
    file: Union[str, WindowsPath], out_file: Union[str, WindowsPath] = None, **kwargs
) -> None:
    """
    Convert csv table to parquet

    Parameters
    ----------
    file : Union[str, WindowsPath]
        Path to csv file to convert
    out_file : Union[str, WindowsPath], optional
        Path to parquet file to be written. If not provided it will use the path of 'file'
    **kwargs
        pandas.read_csv kwargs. See pandas.read_csv documentation

    Raises
    ------
    TypeError
        If 'file' is not a csv file
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
    Convert excel table to parquet

    Parameters
    ----------
    file : Union[str, WindowsPath]
        Path to excel file to convert
    out_file : Union[str, WindowsPath], optional
        Path to parquet file to be written. If not provided it will use the path of 'file'
    **kwargs
        pandas.read_excel kwargs. See pandas.read_excel documentation

    Raises
    ------
    TypeError
        If 'file' is not an xlsx or xls file
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
