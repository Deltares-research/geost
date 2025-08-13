import logging
import re
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import Point


class ColumnInfo(NamedTuple):
    value: str
    unit: str
    name: str
    standard: bool


class CptMeasurementVar(NamedTuple):
    value: Union[int, float, str]
    unit: str
    quantity: str
    reserved: bool


#
COLUMN_DEFS_DATA_BLOCK_CPT = {
    1: ColumnInfo("length", "m", "penetration length", True),
    2: ColumnInfo("qc", "MPa", "measured cone resistance", True),
    3: ColumnInfo("fs", "MPa", "friction resistance", True),
    4: ColumnInfo("rf", "%", "friction number", True),
    5: ColumnInfo("u1", "MPa", "pore pressure u1", True),
    6: ColumnInfo("u2", "MPa", "pore pressure u2", True),
    7: ColumnInfo("u3", "MPa", "pore pressure u3", True),
    8: ColumnInfo("inclination_res", "degrees", "inclination (resultant)", True),
    9: ColumnInfo("inclination_ns", "degrees", "inclination (North-South)", True),
    10: ColumnInfo("inclination_ew", "degrees", "inclination (East-West)", True),
    11: ColumnInfo(
        "corrected_depth", "m", "corrected depth, below fixed surface", True
    ),  # noqa: E501
    12: ColumnInfo("time", "s", "time", True),
    13: ColumnInfo("qt", "MPa", "corrected cone resistance", True),
    14: ColumnInfo("qn", "MPa", "net cone resistance", True),
    15: ColumnInfo("Bq", "", "pore ratio", True),
    16: ColumnInfo("Nm", "", "cone resistance number", True),
    17: ColumnInfo("gamma", "kN/m3", "weight per unit volume", True),
    18: ColumnInfo("u0", "MPa", "in situ, initial pore pressure", True),
    19: ColumnInfo("sigma", "MPa", "total vertical soil pressure", True),
    20: ColumnInfo("sigma_eff", "MPa", "effective vertical soil pressure", True),
    21: ColumnInfo("inclination_x", "degrees", "Inclination in X direction", True),
    22: ColumnInfo("inclination_y", "degrees", "Inclination in Y direction", True),
    23: ColumnInfo("ec", "S/m", "Electric conductivity", True),
    24: ColumnInfo("Bx", "nT", "magnetic field strength in X direction", True),
    25: ColumnInfo("By", "nT", "magnetic field strength in Y direction", True),
    26: ColumnInfo("Bz", "nT", "magnetic field strength in Z direction", True),
    # reserved for future use
    27: ColumnInfo("", "degrees", "magnetic inclination", True),
    # reserved for future use
    28: ColumnInfo("", "degrees", "magnetic inclination", True),
}


RESERVED_MEASUREMENTVARS_CPT = {
    1: CptMeasurementVar(1000, "mm2", "nom. surface area cone tip", True),
    2: CptMeasurementVar(15000, "mm2", "nom. surface area friction sleeve", True),
    3: CptMeasurementVar(None, "", "net surface area quotient of cone tip", True),
    4: CptMeasurementVar(
        None, "", "net surface area quotient of friction sleeve", True
    ),  # noqa: E501
    5: CptMeasurementVar(
        100, "mm", "distance of cone to centre of friction sleeve", True
    ),  # noqa: E501
    6: CptMeasurementVar(None, "", "friction present", True),
    7: CptMeasurementVar(None, "", "PPT u1 present", True),
    8: CptMeasurementVar(None, "", "PPT u2 present", True),
    9: CptMeasurementVar(None, "", "PPT u3 present", True),
    10: CptMeasurementVar(None, "", "inclination measurement present", True),
    11: CptMeasurementVar(None, "", "use of back-flow compensator", True),
    12: CptMeasurementVar(None, "", "type of cone penetration test", True),
    13: CptMeasurementVar(None, "m", "pre-excavated depth", True),
    14: CptMeasurementVar(None, "m", "groundwater level", True),
    15: CptMeasurementVar(None, "m", "water depth (for offshore)", True),
    16: CptMeasurementVar(None, "m", "end depth of penetration test", True),
    17: CptMeasurementVar(None, "", "stop criteria", True),
    # 18: CptMeasurementVar(None, '', 'for future use', True),
    # 19: CptMeasurementVar(None, '', 'for future use', True),
    20: CptMeasurementVar(None, "MPa", "zero measurement cone before", True),
    21: CptMeasurementVar(None, "MPa", "zero measurement cone after", True),
    22: CptMeasurementVar(None, "MPa", "zero measurement friction before", True),
    23: CptMeasurementVar(None, "MPa", "zero measurement friction after", True),
    24: CptMeasurementVar(None, "MPa", "zero measurement PPT u1 before", True),
    25: CptMeasurementVar(None, "MPa", "zero measurement PPT u1 after", True),
    26: CptMeasurementVar(None, "MPa", "zero measurement PPT u2 before", True),
    27: CptMeasurementVar(None, "MPa", "zero measurement PPT u2 after", True),
    28: CptMeasurementVar(None, "MPa", "zero measurement PPT u3 before", True),
    29: CptMeasurementVar(None, "MPa", "zero measurement PPT u3 after", True),
    30: CptMeasurementVar(None, "degrees", "zero measurement inclination before", True),
    31: CptMeasurementVar(None, "degrees", "zero measurement inclination after", True),
    32: CptMeasurementVar(
        None, "degrees", "zero measurement inclination NS before", True
    ),  # noqa: E501
    33: CptMeasurementVar(
        None, "degrees", "zero measurement inclination NS after", True
    ),  # noqa: E501
    34: CptMeasurementVar(
        None, "degrees", "zero measurement inclination EW before", True
    ),  # noqa: E501
    35: CptMeasurementVar(
        None, "degrees", "zero measurement inclination EW after", True
    ),  # noqa: E501
    # 36: CptMeasurementVar(None, '', 'for future use', True),
    # 37: CptMeasurementVar(None, '', 'for future use', True),
    # 38: CptMeasurementVar(None, '', 'for future use', True),
    # 39: CptMeasurementVar(None, '', 'for future use', True),
    # 40: CptMeasurementVar(None, '', 'for future use', True),
    41: CptMeasurementVar(None, "km", "mileage", True),
    42: CptMeasurementVar(
        None, "degrees", "Orientation between X axis inclination and North", True
    ),  # noqa: E501
}


GEF_CPT_REFERENCE_LEVELS = {
    "00000": "own reference level",
    "00001": "Low Low Water Spring",
    "31000": "NAP",
    "32000": "Ostend Level",
    "32001": "TAW",
    "49000": "Normall Null",
}


class CptGefFile:  # TODO: Break parser down in HeaderParser and DataParser and returning a CPT object
    """
    Parse a .gef file of a CPT sounding for its content to retrieve the data and
    relevant information about the measurements.

    Parameters
    ----------
    path : str, Path, optional
        Path to the gef file to parse. If None, an empty class instance is returned.
    sep : str, optional
        Column separator character of the gef file to use when the separator is not
        specified within the gef file header.

    """

    def __init__(self, path: str | Path = None, sep: str = " "):
        self.path = path
        self._header = None
        self._data = None

        self.nr = None
        self.x = None
        self.y = None
        self.z = None
        self.enddepth = None

        # mandatory gef header attributes
        self.gefid = None
        self.ncolumns = None
        self.columninfo = dict()
        self.companyid = None
        self.filedate = None
        self.fileowner = None
        self.lastscan = None
        self.procedurecode = None  # mandatory if gefid is 1, 0, 0
        self.reportcode = None  # this or procedurecode is mandatory if gefid > 1, 1, 0
        self.projectid = None
        self.measurementtext = dict()

        # Additional gef header attributes
        self.columnvoid = dict()
        self.columnminmax = None
        self.columnseparator = sep  # default separator used if not in gef file header
        self.dataformat = None
        self.measurementvars = dict()
        self.recordseparator = None
        self.reportdataformat = None
        self.specimenvars = None
        self.startdate = None
        self.starttime = None
        self.coord_system = None
        self.reference_system = None

        if path:
            self.__open_file(path)

    def __repr__(self):
        return f"{self.__class__.__name__}(nr={self.nr})"

    def __open_file(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            end_header = re.search(r"(?P<eoh>#EOH[=\s+]+)", text).group("eoh")

            self._header, self._data = text.split(end_header)

        self.parse_header()
        self.parse_data()

        self.get_enddepth()

    @property
    def df(self):
        if not hasattr(self, "_df"):
            self.to_df()
        return self._df

    @property
    def header(self):
        header = pd.Series(
            [self.nr, self.x, self.y, self.z, self.enddepth, self.point],
            index=["nr", "x", "y", "z", "enddepth", "geometry"],
        )
        return header

    @property
    def columns(self):
        columns = [f"{c.value}" for c in self.columninfo.values()]
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
            keyword = re.search(r"([#\s]*([A-Z]+)\s*=)\s*", line)

            try:
                keyword_method = keyword.group(2).lower()
            except AttributeError:
                continue

            __method = f"_parse_{keyword_method}"
            if hasattr(self, __method):
                line = line.replace(keyword.group(0), "")
                # remove unnecessary whitespace and string quotes
                line = re.sub(r'["\']|\s\s+', "", line)
                self.__call_header_method(__method, line)

    @staticmethod
    def __split_line(line):
        return [l.strip() for l in line.split(",")]  # noqa: E741

    def parse_data(self):
        """
        Parse datablock of the gef file.

        """
        data = self._data

        if not self.recordseparator:
            r = "!"
        else:
            r = self.recordseparator

        data = data.replace(self.columnseparator, ",")
        data = [self.__split_line(d.rstrip(f",{r}")) for d in data.splitlines()]
        self._data = data

    def to_df(self):
        """
        Create a Pandas DataFrame from the gef datablock.

        Returns
        -------
        None.

        """
        df = pd.DataFrame(self._data)
        df.dropna(inplace=True)
        df = df.astype(float)
        df.replace(self.columnvoid, np.nan, inplace=True)
        df.columns = self.columns

        if "rf" not in df.columns:
            try:
                df["rf"] = (df["fs"] / df["qc"]) * 100
            except KeyError:
                logging.warning(
                    f"Missing data in {self.nr}. Data present: {self.columns}"
                )

        if (
            "corrected_depth" in df.columns
        ):  # TODO: implement calc corrected depth from inclination if not in columns  # noqa: E501
            df["depth"] = df["corrected_depth"]
        else:
            df["depth"] = df["length"]

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
        idx, unit, value, number = self.__split_line(line)
        idx = self.to_zero_indexed(idx)
        info = COLUMN_DEFS_DATA_BLOCK_CPT.get(int(number), "empty")

        if info == "empty":
            logging.warning(f"Unknown information in datablock of {self.path}")
            info = ColumnInfo(value, unit, value, False)

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

    # TODO: check how to fix if zid occurs in header more than once
    def _parse_zid(self, line: str):
        zid = self.__split_line(line)
        if len(zid) == 2:
            reference_system = zid[0]
            self.z = float(zid[1])
            self.delta_z = None
        elif len(zid) == 3:
            reference_system = zid[0]
            self.z = float(zid[1])
            self.delta_z = float(zid[2])
        else:
            logging.warning(
                f"Unclear information in #ZID of {self.path}. "
                "Check zid attribute manually."
            )
            self.zid = zid

        self.reference_system = GEF_CPT_REFERENCE_LEVELS[reference_system]

    # TODO: add correct parsing of reserved measurementtexts
    def _parse_measurementtext(self, line: str):
        text = self.__split_line(line)
        nr, info = int(text[0]), text[1:]
        self.measurementtext.update({nr: info})

    def _parse_xyid(self, line: str):
        xyid = self.__split_line(line)

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
                f"Unclear information in #XYID of {self.path}. "
                "Check xyid attribute manually."
            )
            self.xyid = xyid

    def _parse_columnvoid(self, line: str):
        idx, value = self.__split_line(line)
        idx = self.to_zero_indexed(idx)
        self.columnvoid.update({idx: float(value)})

    def _parse_columnminmax(self, line: str):
        pass

    def _parse_columnseparator(self, line: str):
        self.columnseparator = line

    def _parse_dataformat(self, line: str):
        pass

    def _parse_measurementvar(self, line: str):
        num, val, unit, quantity = self.__split_line(line)

        num = int(num)
        val = safe_float(val)

        _mv = RESERVED_MEASUREMENTVARS_CPT.get(num, "empty")

        if _mv == "empty":
            mvar = CptMeasurementVar(val, unit, quantity, False)
        else:
            if val:
                mvar = CptMeasurementVar(val, _mv.unit, _mv.quantity, True)
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
            d = self.df["length"].max()

        self.enddepth = self.z - d


def safe_float(number):
    try:
        return float(number)
    except ValueError:
        return None
