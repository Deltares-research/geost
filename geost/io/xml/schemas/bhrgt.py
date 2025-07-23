from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "deliveredLocation/bhrgtcom:location/gml:Point/gml:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "deliveredLocation/bhrgtcom:location/gml:Point",
        "resolver": resolvers.parse_crs,
    },
    "surface_level": {
        "xpath": "deliveredVerticalPosition/bhrgtcom:offset",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "vertical_datum": {
        "xpath": "deliveredVerticalPosition/bhrgtcom:verticalDatum",
        "el-attr": "text",
    },
    "groundwater_level": {
        "xpath": "boring/bhrgtcom:groundwaterLevel",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "final_depth": {
        "xpath": "boring/bhrgtcom:finalDepthBoring",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "data": {
        "xpath": "boreholeSampleDescription/bhrgtcom:descriptiveBoreholeLog",
        "resolver": resolvers.process_bhrgt_data,
        "layer-attributes": ["upperBoundary", "lowerBoundary", "geotechnicalSoilName"],
    },
}

WIERTSEMA = {
    "payload_root": "ns1:sourceDocument",
    "nr": {"xpath": "ns1:objectIdAccountableParty"},
    "location": {
        "xpath": "ns1:deliveredLocation/ns3:location/ns2:Point/ns2:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "ns1:deliveredLocation/ns3:location/ns2:Point",
        "resolver": resolvers.parse_crs,
    },
    "surface_level": {
        "xpath": "ns1:deliveredVerticalPosition/ns3:offset",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "vertical_datum": {
        "xpath": "ns1:deliveredVerticalPosition/ns3:verticalDatum",
        "el-attr": "text",
    },
    "groundwater_level": {
        "xpath": "ns1:boring/ns3:groundwaterLevel",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "final_depth": {
        "xpath": "ns1:boring/ns3:finalDepthBoring",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "data": {
        "xpath": "ns1:boreholeSampleDescription/ns3:descriptiveBoreholeLog",
        "resolver": resolvers.process_bhrgt_data,
        "layer-attributes": ["upperBoundary", "lowerBoundary", "geotechnicalSoilName"],
    },
}

SCHEMA = {"BRO": BRO, "Wiertsema": WIERTSEMA}
