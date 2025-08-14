from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "deliveredLocation/bhrcommon:location/gml:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "deliveredLocation/bhrcommon:location",
        "resolver": resolvers.parse_crs,
    },
    "surface": {
        "xpath": "deliveredVerticalPosition/bhrcommon:offset",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "vertical_datum": {
        "xpath": "deliveredVerticalPosition/bhrcommon:verticalDatum",
        "el-attr": "text",
    },
    "begin_depth": {
        "xpath": "boring/bhrcommon:boredTrajectory/bhrcommon:beginDepth",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "end": {
        "xpath": "boring/bhrcommon:boredTrajectory/bhrcommon:endDepth",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "ghg": {
        "xpath": "boreholeSampleDescription/bhrcommon:result/bhrcommon:meanHighestGroundwaterLevel",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "glg": {
        "xpath": "siteCharacteristic/bhrcommon:meanLowestGroundwaterTable",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "landuse": {"xpath": "siteCharacteristic/bhrcommon:landUse"},
    "data": {
        "xpath": "boreholeSampleDescription/bhrcommon:result",
        "resolver": resolvers.process_bhrp_data,
        "layer-attributes": ["upperBoundary", "lowerBoundary", "standardSoilName"],
    },
}

schemas = {"BRO": BRO}
