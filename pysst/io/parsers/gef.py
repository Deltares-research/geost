# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:07:08 2023

@author: knaake
"""
import re
import logging
from collections import namedtuple
from pathlib import Path, WindowsPath


columninfo = namedtuple('columninfo', 'value unit name')

column_defs_data_block = {
    1: columninfo('length', 'm', 'penetration length'),
    2: columninfo('qc', 'MPa', 'measured cone resistance'),
    3: columninfo('fs', 'MPa', 'friction resistance'),
    4: columninfo('rf', '%', 'friction number'),
    5: columninfo('u1', 'MPa', 'pore pressure u1'),
    6: columninfo('u2', 'MPa', 'pore pressure u2'),
    7: columninfo('u3', 'MPa', 'pore pressure u3'),
    8: columninfo('inclination_res', 'degrees', 'inclination (resultant)'),
    9: columninfo('inclination_ns', 'degrees', 'inclination (North-South)'),
    10: columninfo('inclination_ew', 'degrees', 'inclination (East-West)'),
    11: columninfo('length_corr', 'm', 'corrected depth, below fixed surface'),
    12: columninfo('time', 's', 'time'),
    13: columninfo('qt', 'MPa', 'corrected cone resistance'),
    14: columninfo('qn', 'MPa', 'net cone resistance'),
    15: columninfo('Bq', '', 'pore ratio'),
    16: columninfo('Nm', '', 'cone resistance number'),
    17: columninfo('gamma', 'kN/m3', 'weight per unit volume'),
    18: columninfo('u0', 'MPa', 'in situ, initial pore pressure'),
    19: columninfo('sigma', 'MPa', 'total vertical soil pressure'),
    20: columninfo('sigma_eff', 'MPa', 'effective vertical soil pressure'),
    21: columninfo('inclination_x', 'degrees', 'Inclination in X direction'),
    22: columninfo('inclination_y', 'degrees', 'Inclination in Y direction'),
    23: columninfo('ec', 'S/m', 'Electric conductivity'),
    24: columninfo('Bx', 'nT', 'magnetic field strength in X direction'),
    25: columninfo('By', 'nT', 'magnetic field strength in Y direction'),
    26: columninfo('Bz', 'nT', 'magnetic field strength in Z direction'),
    27: columninfo('', 'degrees', 'magnetic inclination'), # reserved for future use
    28: columninfo('', 'degrees', 'magnetic inclination'), # reserved for future use
    }


class GefFile:
    
    def __init__(self, path: str | WindowsPath, sep: str = ';'):
        self.path = path
        self._header = None
        self._data = None
        
        ## File information attributes        
        self.sep = sep
        self.ncolumns = None
        self.data_columns = dict()
        self.nodata_vals = dict()
        
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
            if '#GEFID=' in line:
                pass
            
            elif '#COLUMN=' in line:
                self.ncolumns = __parse_ncolumn_line(line)
            
            elif '#COLUMNINFO=' in line:
                idx, info = self.__parse_columninfo_line(line)
                self.data_columns.update({idx: info})
            
            elif '#COLUMNMINMAX=' in line:
                pass
            
            elif '#COLUMNSEPARATOR=' in line:
                pass
            
            elif '#COLUMNTEXT=' in line:
                pass
                
            elif '#COLUMNVOID=' in line:
                pass
            
            elif '#COMPANYID=' in line:
                pass
            
            elif '#DATAFORMAT=' in line:
                pass
            
            elif '#FILEDATE=' in line:
                pass
            
            elif '#FILEOWNER=' in line:
                pass
            
            elif '#LASTSCAN=' in line:
                pass
            
            elif '#MEASUREMENTTEXT=' in line:
                pass
            
            elif '#MEASUREMENTVAR=' in line:
                pass
            
            elif '#PROCEDURECODE=' in line:
                pass
            
            elif '#REPORTCODE=' in line:
                pass
            
            elif '#PROJECTID=' in line:
                pass
            
            elif '#RECORDSEPARATOR=' in line:
                pass
            
            elif '#REPORTDATAFORMAT=' in line:
                pass
            
            elif '#SPECIMENVAR=' in line:
                pass
            
            elif '#STARTDATE=' in line:
                pass
            
            elif '#STARTTIME=' in line:
                pass
            
            elif '#TESTID=' in line:
                pass
            
            elif '#XYID=' in line:
                pass
            
            elif '#ZID=' in line:
                pass
        
        return header
    
    def __parse_columninfo_line(self, line):
        line = re.sub(r"#COLUMNINFO=\s*", '', line)
        idx, unit, value, number = line.split(', ')
        
        info = column_defs_data_block.get(int(number), 'empty')
        
        if info == 'empty':
            logging.warning(f'Unknown information in datablock of {self.path}')
        
        return int(idx), info
    
    def parse_data(self):
        data = self._data.splitlines()
        data = [d.split(';') for d in data]
        return data


def __parse_ncolumn_line(line):
    ncolumns = re.sub(r"#COLUMN=\s*", '', line)
    return int(ncolumns)


if __name__ == "__main__":
    workdir = Path(r'n:\My Documents\margriettunnel\data\cpts')
    file = workdir/r'CPT000000157983_IMBRO.gef'
    gef = GefFile(file)
    
    for line in gef._header:
        if '#COLUMN=' in line:
            print(line)
            
            