from typing import Iterable
from pathlib import Path, WindowsPath
from pysst.utils import get_path_iterable
from pysst.io.parsers.gef_parsers import CptGefFile


def read_cpt_gef_files(file_or_folder):
    
    if isinstance(file_or_folder, (str, WindowsPath)):
        files = get_path_iterable(Path(file_or_folder))
    
    elif isinstance(file_or_folder, Iterable):
        files = file_or_folder
    
    for f in files:
        cpt = CptGefFile(f)
        df = cpt.df
        df.insert(0, 'nr', cpt.nr)
        df.insert(1, 'x', cpt.x)
        df.insert(2, 'y', cpt.y)
        df.insert(3, 'z', cpt.z)
        df.insert(4, 'end', cpt.enddepth)
        yield df