from enum import Enum
import pandas as pd
from pygef import Cpt

from pysst.utils import get_path_iterable


def pygef_gef_cpt(file_or_folder):
    """
    Use the pygef GEF reader to generate a pysst-compatible pandas.DataFrame

    Parameters
    ----------
    file_or_folder : Union[str, WindowsPath]
        gef file or folder containing gef files

    Yields
    ------
    pd.DataFrame
        pd.DataFrame per cpt
    """
    for gef_file in get_path_iterable(file_or_folder, wildcard="*.gef"):
        gef_cpt = Cpt(str(gef_file))
        gef_cpt_df = gef_cpt.df.to_pandas()
        gef_id = gef_cpt.test_id
        gef_x = gef_cpt.x
        gef_y = gef_cpt.y
        gef_mv = gef_cpt.df["elevation_with_respect_to_nap"][0]
        gef_end = (
            gef_cpt.df["elevation_with_respect_to_nap"][0]
            - gef_cpt.df["corrected_depth"][-1]
        )
        gef_top = gef_cpt_df["elevation_with_respect_to_nap"]
        gef_bottom = (
            gef_cpt_df["elevation_with_respect_to_nap"][0]
            - gef_cpt_df["corrected_depth"]
        )
        extra_cols = pd.DataFrame(
            {
                "nr": [gef_id for i in range(len(gef_cpt_df))],
                "x": [gef_x for i in range(len(gef_cpt_df))],
                "y": [gef_y for i in range(len(gef_cpt_df))],
                "mv": [gef_mv for i in range(len(gef_cpt_df))],
                "end": [gef_end for i in range(len(gef_cpt_df))],
                "top": gef_top,
                "bottom": gef_bottom,
            }
        )
        yield extra_cols.join(gef_cpt_df)
