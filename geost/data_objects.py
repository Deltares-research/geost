from dataclasses import dataclass


@dataclass
class Cpt:
    nr: str = None
    x: int | float = None
    y: int | float = None
    z: int | float = None
    enddepth: int | float = None
    gefid = None
    ncolumns = None
    columninfo: dict = None
    companyid = None
    filedate = None
    fileowner = None
    lastscan = None
    procedurecode = None  # mandatory if gefid is 1, 0, 0
    reportcode = None  # this or procedurecode is mandatory if gefid > 1, 1, 0
    projectid = None
    measurementtext: dict = None
    columnvoid: dict = None
    columnminmax = None
    dataformat = None
    measurementvars: dict = None
    recordseparator = None
    reportdataformat = None
    specimenvars = None
    startdate = None
    starttime = None
    coord_system = None
    reference_system = None

    def __post_init__(self):
        pass
