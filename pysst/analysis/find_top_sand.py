# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:28:57 2022

@author: knaake
"""
import numpy as np


def find_top_sand(lith, top, bottom, min_sand_frac, min_sand_thickness):
    """
    Find the top of sand depth in a borehole. The top of sand is defined by the
    first layer of a specified thickness that contains a minimum percentage of
    sand. By default: when the first layer of sand is detected, the next 1 meter
    is scanned. Within this meter, if more than 50% of the lenght has a main
    lithology of sand, the initially detected layer of sand is regarded as the top
    of sand. If not, continue downward until the next layer of sand is detected and repeat.

    Parameters
    ----------
    lith : ndarray
        Numpy array containing the lithology of the borehole.
    top : ndarray
        Numpy array containing the top depth of the layers of the borehole.
    bottom : ndarray
        Numpy array containing the bottom depth of the layers of the borehole.
    min_sand_frac : float
        Minimum percentage required to be sand.
    min_sand_thickness : float
        Minimum thickness of the sand to search for.

    Returns
    -------
    top_sand : float
        Top depth of the sand layer that meets the requirements.

    """
    is_sand = lith == "Z"

    if np.any(is_sand):
        idx_sand = np.where(is_sand)[0]
        for idx in idx_sand:
            top_sand = top[idx]
            search_depth = top_sand - min_sand_thickness

            search_mask = (top <= top_sand) & (top > search_depth)

            tmp_top = top[search_mask]
            tmp_bottom = bottom[search_mask]
            tmp_bottom[-1] = search_depth

            length = tmp_top - tmp_bottom

            sand_frac = length[is_sand[search_mask]].sum() / min_sand_thickness

            if sand_frac >= min_sand_frac:
                break

    else:
        top_sand = np.nan

    return top_sand


def top_of_sand(boreholes, ids="nr", min_sand_frac=0.5, min_sand_thickness=1):

    groupby = boreholes.groupby(ids)

    for nr, df in groupby:
        lith = df["lith"].values
        top = df["top"].values
        bottom = df["bottom"].values

        top_sand = find_top_sand(lith, top, bottom, min_sand_frac, min_sand_thickness)

        yield (nr, top_sand)
