# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:03:52 2023

@author: knaake
"""
from collections import namedtuple


rdcoord = namedtuple('RD', 'x y epsg')
ddcoord = namedtuple('DecimalDegree', 'lat lon epsg')

layer_info = (
    'top '
    'bottom '
    'horizon '
    'standard_name '
    'pedologic_name '
    'om '
    'ca '
    'gravel '
    'shells '
    'clay_perc'
    )

LayerSoilCore = namedtuple('Layer', layer_info)