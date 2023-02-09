# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:07:08 2023

@author: knaake
"""
import re
import logging
import time
import numpy as np
from pygef import Cpt
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path, WindowsPath


columninfo = namedtuple('columninfo', 'value unit name standard')

column_defs_data_block = {
    1: columninfo('length', 'm', 'penetration length', True),
    2: columninfo('qc', 'MPa', 'measured cone resistance', True),
    3: columninfo('fs', 'MPa', 'friction resistance', True),
    4: columninfo('rf', '%', 'friction number', True),
    5: columninfo('u1', 'MPa', 'pore pressure u1', True),
    6: columninfo('u2', 'MPa', 'pore pressure u2', True),
    7: columninfo('u3', 'MPa', 'pore pressure u3', True),
    8: columninfo('inclination_res', 'degrees', 'inclination (resultant)', True),
    9: columninfo('inclination_ns', 'degrees', 'inclination (North-South)', True),
    10: columninfo('inclination_ew', 'degrees', 'inclination (East-West)', True),
    11: columninfo('length_corr', 'm', 'corrected depth, below fixed surface', True),
    12: columninfo('time', 's', 'time', True),
    13: columninfo('qt', 'MPa', 'corrected cone resistance', True),
    14: columninfo('qn', 'MPa', 'net cone resistance', True),
    15: columninfo('Bq', '', 'pore ratio', True),
    16: columninfo('Nm', '', 'cone resistance number', True),
    17: columninfo('gamma', 'kN/m3', 'weight per unit volume', True),
    18: columninfo('u0', 'MPa', 'in situ, initial pore pressure', True),
    19: columninfo('sigma', 'MPa', 'total vertical soil pressure', True),
    20: columninfo('sigma_eff', 'MPa', 'effective vertical soil pressure', True),
    21: columninfo('inclination_x', 'degrees', 'Inclination in X direction', True),
    22: columninfo('inclination_y', 'degrees', 'Inclination in Y direction', True),
    23: columninfo('ec', 'S/m', 'Electric conductivity', True),
    24: columninfo('Bx', 'nT', 'magnetic field strength in X direction', True),
    25: columninfo('By', 'nT', 'magnetic field strength in Y direction', True),
    26: columninfo('Bz', 'nT', 'magnetic field strength in Z direction', True),
    27: columninfo('', 'degrees', 'magnetic inclination', True), # reserved for future use
    28: columninfo('', 'degrees', 'magnetic inclination', True), # reserved for future use
    }


class GefFile:
    
    def __init__(self, path: str | WindowsPath, sep: str = ';'):
        self.path = path
        self._header = None
        self._data = None
        
        self.nr = None
        self.x = None
        self.y = None
        self.z = None
        
        ## mandatory gef header attributes
        self.gefid = None
        self.column = None
        self.columninfo = dict()
        self.companyid = None
        self.filedate = None
        self.fileowner = None
        self.lastscan = None
        self.procedurecode = None # mandatory if gefid is 1, 0, 0
        self.reportcode = None # either this or procedurecode is mandatory if gefid is 1, 1, 0 or higher
        self.projectid = None
        self.measurementtext = dict()
        
        ## Additional gef header attributes
        self.columnvoid = dict()
        self.columnminmax = None
        self.columnseparator = sep # default value used if not in gef file header
        self.dataformat = None
        self.measurementvars = dict()
        self.recordseparator = None
        self.reportdataformat = None
        self.specimenvars = None
        self.startdate = None
        self.starttime = None
        self.coord_system = None
        self.reference_system = None
        
        self.__open_file(path)
        
    def __open_file(self, path):
        with open(path, 'r') as f:
            text = f.read()
            end_header = re.search(r'(?P<eoh>#EOH[=\s+]+)', text).group('eoh')
            
            self._header, self._data = text.split(end_header)
        
        self.parse_header()
        self.parse_data()
    
    def parse_header(self):
        header = self._header.splitlines()
        
        for line in header:
            keyword = re.match(r'([#\s]*([A-Z]+)\s*=)\s*', line)
            keyword_method = keyword.group(2).lower()
            
            if hasattr(self, f'_parse_{keyword_method}'):
                line = line.lstrip(keyword.group(0))
                self.__call_header_method(keyword_method, line)
        
        return header
    
    def parse_data(self):
        data = self._data.splitlines()
        data = [d.split(self.columnseparator) for d in data]
        return data
    
    def __call_header_method(self, method, line):
        return getattr(self, f'_parse_{method}')(line)
    
    def _parse_gefid(self, line):
        self.gefid = line
        
    def _parse_column(self, line):
        self.column = int(line)
        
    def _parse_columninfo(self, line: str):
        """
        Parse #COLUMNINFO line in the gef file.

        """
        idx, unit, value, number = line.split(', ')
        info = column_defs_data_block.get(int(number), 'empty')
        
        if info == 'empty':
            logging.warning(f'Unknown information in datablock of {self.path}')
            info = columninfo(value, unit, value, False)
        
        self.columninfo.update({int(idx): info})
    
    def _parse_companyid(self, line: str):
        self.companyid = line
    
    def _parse_filedate(self, line: str):
        pass
    
    def _parse_fileowner(self, line: str):
        self.fileowner = line
    
    def _parse_lastscan(self, line: str):
        self.lastscan = int(line)
    
    def _parse_procedurecode(self, line: str):
        self.procedurecode = line
    
    def _parse_reportcode(self, line: str):
        self.reportcode = line
    
    def _parse_projectid(self, line: str):
        self.projectid = line
    
    def _parse_testid(self, line: str):
        """
        Get the testid number (nr) from the gef file.

        """
        self.nr = line
    
    def _parse_zid(self, line: str): # TODO: check how to fix if zid occurs in header more than once
        zid = line.split(', ')
        if len(zid) == 2:
            self.reference_system = zid[0]
            self.z = float(zid[1])
        elif len(zid) == 3:
            self.reference_system = zid[0]
            self.z = float(zid[1])
            self.delta_z = float(zid[2])
        else:
            logging.warning(
                f'Unclear information in #ZID of {self.path}. '
                'Check zid attribute manually.'
                )
            self.zid = zid
    
    def _parse_measurementtext(self, line: str):
        test = line.split(', ')
        print(len(test))
    
    def _parse_xyid(self, line: str):
        xyid = line.split(', ')
        
        if len(xyid) == 3:
            self.coord_system = xyid[0]
            self.x = float(xyid[1])
            self.y = float(xyid[2])
        
        elif len(xyid) == 5:
            self.coord_system = xyid[0]
            self.x = float(xyid[1])
            self.y = float(xyid[2])
            self.dx = float(xyid[3])
            self.dy = float(xyid[4])
        
        else:
            logging.warning(
                f'Unclear information in #XYID of {self.path}. '
                'Check xyid attribute manually.'
                )
            self.xyid = xyid
    
    def _parse_columnvoid(self, line: str):
        idx, value = line.split(', ')
        self.columnvoid.update({int(idx): float(value)})
    
    def _parse_columnminmax(self, line: str):
        pass
    
    def _parse_columnseparator(self, line: str):
        pass
    
    def _parse_dataformat(self, line: str):
        pass
    
    def _parse_measurementvar(self, line: str):
        pass
    
    def _parse_recordseparator(self, line: str):
        pass
    
    def _parse_reportdataformat(self, line: str):
        pass
    
    def _parse_specimenvar(self, line: str):
        pass
    
    def _parse_startdate(self, line: str):
        pass
    
    def _parse_starttime(self, line: str):
        pass


if __name__ == "__main__":
    workdir = Path(r'n:\My Documents\margriettunnel\data\cpts')
    file = workdir/r'CPT000000157983_IMBRO.gef'
    # file = workdir/r'83268_DKMP001-A_(DKMP_C01).GEF'
    
    new_reader = []
    pygef = []
    
    # for i in tqdm(range(100), total=100):
    start = time.time()
    gef = GefFile(file)
    end = time.time()
    # new_reader.append(end-start)
    
    # print(gef.nr, gef.coord_system, gef.x, gef.y, gef.reference_system, gef.z)
    
    # start = time.time()
    # gef = Cpt(str(file))
    # end = time.time()
    # pygef.append(end-start)
        
    # print('\n')
    # print(f"New reader took {np.mean(new_reader)} seconds")
    # print(f"Pygef took {np.mean(pygef)} seconds")

    # for line in gef._header.splitlines():
    #     keyword = re.match(r'([#\s]*([A-Z]+)\s*=)\s*', line)
        
    #     keyword_method = keyword.group(2).lower()
    #     if keyword_method == 'zid':
    #         zid = line.lstrip(keyword.group(0))
            
        
        
            
            