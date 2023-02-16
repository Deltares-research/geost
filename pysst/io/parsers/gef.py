import re
import logging
import time
import numpy as np
import pandas as pd
from pygef import Cpt
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path, WindowsPath
from shapely.geometry import Point
from pysst.borehole import CptCollection
from pysst.utils import safe_float, get_path_iterable
from typing import Iterable


columninfo = namedtuple('columninfo', ['value', 'unit', 'name', 'standard'])
measurementvar = namedtuple('measurementvar', ['value', 'unit', 'quantity', 'reserved'])

column_defs_data_block_cpt = {
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
    11: columninfo('corrected_depth', 'm', 'corrected depth, below fixed surface', True),
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


reserved_measurementvars_cpt = {
    1: measurementvar(1000, 'mm2', 'nom. surface area cone tip', True),
    2: measurementvar(15000, 'mm2', 'nom. surface area friction sleeve', True),
    3: measurementvar(None, '', 'net surface area quotient of cone tip', True),
    4: measurementvar(None, '', 'net surface area quotient of friction sleeve', True),
    5: measurementvar(100, 'mm', 'distance of cone to centre of friction sleeve', True),
    6: measurementvar(None, '', 'friction present', True),
    7: measurementvar(None, '', 'PPT u1 present', True),
    8: measurementvar(None, '', 'PPT u2 present', True),
    9: measurementvar(None, '', 'PPT u3 present', True),
    10: measurementvar(None, '', 'inclination measurement present', True),
    11: measurementvar(None, '', 'use of back-flow compensator', True),
    12: measurementvar(None, '', 'type of cone penetration test', True),
    13: measurementvar(None, 'm', 'pre-excavated depth', True),
    14: measurementvar(None, 'm', 'groundwater level', True),
    15: measurementvar(None, 'm', 'water depth (for offshore)', True),
    16: measurementvar(None, 'm', 'end depth of penetration test', True),
    17: measurementvar(None, '', 'stop criteria', True),
    # 18: measurementvar(None, '', 'for future use', True),
    # 19: measurementvar(None, '', 'for future use', True),
    20: measurementvar(None, 'MPa', 'zero measurement cone before', True),
    21: measurementvar(None, 'MPa', 'zero measurement cone after', True),
    22: measurementvar(None, 'MPa', 'zero measurement friction before', True),
    23: measurementvar(None, 'MPa', 'zero measurement friction after', True),
    24: measurementvar(None, 'MPa', 'zero measurement PPT u1 before', True),
    25: measurementvar(None, 'MPa', 'zero measurement PPT u1 after', True),
    26: measurementvar(None, 'MPa', 'zero measurement PPT u2 before', True),
    27: measurementvar(None, 'MPa', 'zero measurement PPT u2 after', True),
    28: measurementvar(None, 'MPa', 'zero measurement PPT u3 before', True),
    29: measurementvar(None, 'MPa', 'zero measurement PPT u3 after', True),
    30: measurementvar(None, 'degrees', 'zero measurement inclination before', True),
    31: measurementvar(None, 'degrees', 'zero measurement inclination after', True),
    32: measurementvar(None, 'degrees', 'zero measurement inclination NS before', True),
    33: measurementvar(None, 'degrees', 'zero measurement inclination NS after', True),
    34: measurementvar(None, 'degrees', 'zero measurement inclination EW before', True),
    35: measurementvar(None, 'degrees', 'zero measurement inclination EW after', True),
    # 36: measurementvar(None, '', 'for future use', True),
    # 37: measurementvar(None, '', 'for future use', True),
    # 38: measurementvar(None, '', 'for future use', True),
    # 39: measurementvar(None, '', 'for future use', True),
    # 40: measurementvar(None, '', 'for future use', True),
    41: measurementvar(None, 'km', 'mileage', True),
    42: measurementvar(None, 'degrees', 'Orientation between X axis inclination and North', True),
    }


gef_cpt_reference_levels = {
    '00000': 'own reference level',
    '00001': 'Low Low Water Spring',
    '31000': 'NAP',
    '32000': 'Ostend Level',
    '32001': 'TAW',
    '49000': 'Normall Null'
    }


class CptGefFile:
    
    def __init__(self, path: str | WindowsPath, sep: str = ';'):
        self.path = path
        self._header = None
        self._data = None
        
        self.nr = None
        self.x = None
        self.y = None
        self.z = None
        self.enddepth = None
        
        ## mandatory gef header attributes
        self.gefid = None
        self.ncolumns = None
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
        self.columnseparator = sep # default separator used if not in gef file header
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
        # self.to_df()
    
    @property
    def df(self):
        if not hasattr(self, '_df'):
            self.to_df()
        return self._df
    
    @property
    def header(self):
        header = pd.Series(
            [self.nr, self.x, self.y, self.z, self.get_enddepth(), self.point],
            index=['nr', 'x', 'y', 'z', 'enddepth', 'geometry']
            )
        return header
    
    @property
    def columns(self):
        columns = [f'{c.value}' for c in self.columninfo.values()]
        return columns
    
    @property
    def point(self):
        return Point(self.x, self.y)
    
    @staticmethod
    def to_zero_indexed(idx: str):
        """
        Update an index from the gef file (is 1-indexed) to a 0-indexed Python
        index.

        """
        return int(idx) - 1
    
    def parse_header(self):
        header = self._header.splitlines()
        
        for line in header:
            keyword = re.match(r'([#\s]*([A-Z]+)\s*=)\s*', line)
            keyword_method = keyword.group(2).lower()
            
            __method = f'_parse_{keyword_method}'
            if hasattr(self, __method):
                line = line.replace(keyword.group(0), '')
                self.__call_header_method(__method, line)
    
    def parse_data(self):
        """
        Parse datablock of the gef file.

        """
        data = self._data.splitlines()
        end_row = self.columnseparator + self.recordseparator
        data = [d.rstrip(end_row).split(self.columnseparator) for d in data]
        self._data = data
        
    def to_df(self):
        """
        Create a Pandas DataFrame from the gef datablock.

        Returns
        -------
        None.

        """
        df = pd.DataFrame(self._data, dtype='float64')
        df.replace(self.columnvoid, np.nan, inplace=True)
        df.columns = self.columns
        
        if 'rf' not in df.columns:
            df['rf'] = (df['fs']/df['qc']) * 100
        
        if 'corrected_depth' in df.columns: # TODO: implement calc corrected depth from inclination if not in columns
            df['depth'] = self.z - df['corrected_depth']
        else:
            df['depth'] = self.z - df['length']
        
        self._df = df
    
    def __call_header_method(self, method, line):
        """
        Helper method to call the correct parser method of the class for a specific
        header attribute (e.g. _parse_columninfo).

        Parameters
        ----------
        method : str
            Name of the header attribute.
        line : str
            Line in the header to parse.

        """
        return getattr(self, method)(line)
    
    def _parse_gefid(self, line):
        self.gefid = line
        
    def _parse_column(self, line):
        self.ncolumns = int(line)
        
    def _parse_columninfo(self, line: str):
        idx, unit, value, number = line.split(', ')
        idx = self.to_zero_indexed(idx)
        info = column_defs_data_block_cpt.get(int(number), 'empty')
        
        if info == 'empty':
            logging.warning(f'Unknown information in datablock of {self.path}')
            info = columninfo(value, unit, value, False)
        
        self.columninfo.update({idx: info})
    
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
        self.nr = line
    
    def _parse_zid(self, line: str): # TODO: check how to fix if zid occurs in header more than once
        zid = line.split(', ')
        if len(zid) == 2:
            reference_system = zid[0]
            self.z = float(zid[1])
        elif len(zid) == 3:
            reference_system = zid[0]
            self.z = float(zid[1])
            self.delta_z = float(zid[2])
        else:
            logging.warning(
                f'Unclear information in #ZID of {self.path}. '
                'Check zid attribute manually.'
                )
            self.zid = zid
        
        self.reference_system = gef_cpt_reference_levels[reference_system]
    
    def _parse_measurementtext(self, line: str): #TODO: add correct parsing of reserved measurementtexts
        text = line.split(', ')
        nr, info = int(text[0]), text[1:]
        self.measurementtext.update({nr: info})
    
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
        idx = self.to_zero_indexed(idx)
        self.columnvoid.update({idx: float(value)})
    
    def _parse_columnminmax(self, line: str):
        pass
    
    def _parse_columnseparator(self, line: str):
        self.columnseparator = line
    
    def _parse_dataformat(self, line: str):
        pass
    
    def _parse_measurementvar(self, line: str):
        num, val, unit, quantity = line.split(', ')
        
        num = int(num)
        val = safe_float(val)
        
        _mv = reserved_measurementvars_cpt.get(num, 'empty')
        
        if _mv == 'empty':
            mvar = measurementvar(val, unit, quantity, False)
        else:
            if val:
                mvar = measurementvar(val, _mv.unit, _mv.quantity, True)
            else:
                mvar = _mv
        
        self.measurementvars.update({num: mvar})
    
    def _parse_recordseparator(self, line: str):
        self.recordseparator = line
    
    def _parse_reportdataformat(self, line: str):
        pass
    
    def _parse_specimenvar(self, line: str):
        pass
    
    def _parse_startdate(self, line: str):
        pass
    
    def _parse_starttime(self, line: str):
        pass
    
    def get_enddepth(self):
        enddepth = self.measurementvars.get(16)
        if enddepth:
            d = enddepth.value
        else:
            d = self.df['length'].max()
        return d
        

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
        

if __name__ == "__main__":
    workdir = Path(r'n:\My Documents\margriettunnel\data\cpts')
    file = workdir/r'CPT000000157983_IMBRO.gef'
    file = workdir/r'83268_DKMP001-A_(DKMP_C01).GEF'
    
    gef = CptGefFile(file)
    # print(gef.header)
    # print(gef.df)
    
    a = pd.concat(read_cpt_gef_files(workdir), ignore_index=True)
    print(a)
    
    for line in gef._header.splitlines():
        keyword = re.match(r'([#\s]*([A-Z]+)\s*=)\s*', line)
        
        keyword_method = keyword.group(2).lower()
        if keyword_method == 'testid':
            break
            a = line.lstrip(keyword.group(0))
            print(a)

    
    #%% Benchmark 100 runs
    
    new_reader = []
    pygef = []
    
    for i in tqdm(range(100), total=100):
        start = time.time()
        gef = CptGefFile(file)
        df = gef.df
        end = time.time()
        new_reader.append(end-start)
        
        start = time.time()
        pgef = Cpt(str(file))
        pdf = pgef.df
        end = time.time()
        pygef.append(end-start)
        
    print('\n')
    print(f"New reader took {np.mean(new_reader)} seconds")
    print(f"Pygef took {np.mean(pygef)} seconds")
            