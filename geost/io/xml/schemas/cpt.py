from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "deliveredLocation/cptcommon:location/gml:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "deliveredLocation/cptcommon:location",
        "resolver": resolvers.parse_crs,
    },
    "surface": {
        "xpath": "deliveredVerticalPosition/cptcommon:offset",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "vertical_datum": {
        "xpath": "deliveredVerticalPosition/cptcommon:verticalDatum",
        "el-attr": "text",
    },
    "predrilled_depth": {
        "xpath": "conePenetrometerSurvey/cptcommon:trajectory/cptcommon:predrilledDepth",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "end": {
        "xpath": "conePenetrometerSurvey/cptcommon:trajectory/cptcommon:finalDepth",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "data": {
        "xpath": "conePenetrometerSurvey",
        "resolver": resolvers.process_cpt_data,
    },
}

schemas = {"BRO": BRO}
