from enum import Enum

# Enum classes for different readers that we are going to implement


class CptXmlReaders(Enum):
    geolib = "geolib"
    pygef = "pygef"


class BroBoreholeReaders(Enum):
    geolib = "geolib"
    pygef = "pygef"
    xsboringen = "xsboringen"
