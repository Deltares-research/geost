from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "deliveredLocation/sfrcom:location/gml:Point/gml:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "deliveredLocation/sfrcom:location/gml:Point",
        "resolver": resolvers.parse_crs,
    },
    "surface": {
        "xpath": "deliveredVerticalPosition/sfrcom:offset",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "vertical_datum": {
        "xpath": "deliveredVerticalPosition/sfrcom:verticalDatum",
        "el-attr": "text",
    },
    "landuse": {
        "xpath": "siteCharacteristic/sfrcom:SiteCharacteristic/sfrcom:soilUse",
        "el-attr": "text",
    },
    "outcrop_type": {
        "xpath": "soilUncovering/sfrcom:SoilUncovering/sfrcom:outcropType",
        "el-attr": "text",
    },
    "end": {
        "xpath": "soilUncovering/sfrcom:SoilUncovering/sfrcom:endDepthSoilFace",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "data": {
        "xpath": "soilFaceDescription/sfrcom:SoilFaceDescription/sfrcom:soilProfile/sfrcom:SoilProfile",
        "resolver": resolvers.process_sfr_data,
        "layer-attributes": ["upperBoundary", "lowerBoundary", "soilNameNEN5104"],
    },
}

schemas = {"BRO": BRO}
