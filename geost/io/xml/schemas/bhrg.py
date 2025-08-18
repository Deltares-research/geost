from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "deliveredLocation/bhrgcom:location/gml:Point/gml:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "deliveredLocation/bhrgcom:location/gml:Point",
        "resolver": resolvers.parse_crs,
    },
    "surface": {
        "xpath": "deliveredVerticalPosition/bhrgcom:offset",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "vertical_datum": {
        "xpath": "deliveredVerticalPosition/bhrgcom:verticalDatum",
        "el-attr": "text",
    },
    "end": {
        "xpath": "boring/bhrgcom:Boring/bhrgcom:finalDepthBoring",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "data": {
        "xpath": "boreholeSampleDescription/bhrgcom:BoreholeSampleDescription/bhrgcom:descriptiveBoreholeLog/bhrgcom:DescriptiveBoreholeLog",
        "resolver": resolvers.process_bhrgt_data,
        "layer-attributes": ["upperBoundary", "lowerBoundary", "soilNameNEN5104"],
    },
}

schemas = {"BRO": BRO}
