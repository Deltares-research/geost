from geost.data import nllat

VERTICAL_DATUM_CONVERSIONS = {
    # From NAP to others
    "EPSG:5709>EPSG:5714": 0,
    "EPSG:5709>EPSG:5710": 2.33,
    "EPSG:5709>EPSG:5861": nllat() * 0.01,
    # From TAW to others
    "EPSG:5710>EPSG:5709": -2.33,
    "EPSG:5710>EPSG:5714": -2.33,
    "EPSG:5710>EPSG:5861": nllat() * 0.01 - 2.33,
    # From MSL to others
    "EPSG:5714>EPSG:5709": 0,
    "EPSG:5714>EPSG:5710": 2.33,
    "EPSG:5714>EPSG:5861": nllat() * 0.01,
    # From LAT to others
    "EPSG:5861>EPSG:5714": nllat() * -0.01,
    "EPSG:5861>EPSG:5709": nllat() * -0.01,
    "EPSG:5861>EPSG:5710": nllat() * -0.01 + 2.33,
}
