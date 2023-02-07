# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:03:52 2023

@author: knaake
"""
from collections import namedtuple


rdcoord = namedtuple('RD', 'x y epsg')
ddcoord = namedtuple('DecimalDegree', 'lat lon epsg')


