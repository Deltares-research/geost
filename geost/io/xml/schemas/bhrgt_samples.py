from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "standardizedLocation/brocom:location/gml:pos",
        "resolver": resolvers.parse_coordinates,
        "el-attr": "text",
    },
    "crs": {
        "xpath": "standardizedLocation/brocom:location",
        "resolver": resolvers.parse_crs,
    },
    "surface": {
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
    "end": {
        "xpath": "boring/bhrgtcom:finalDepthBoring",
        "resolver": resolvers.safe_float,
        "el-attr": "text",
    },
    "data": {
        "xpath": "boreholeSampleAnalysis",
        "resolver": resolvers.process_bhrgt_grain_data,
        "layer-attributes": [
            "beginDepth",
            "endDepth",
            "fractionSmaller63um",
            "fractionLarger63um",
            "fraction63to90um",
            "fraction90to125um",
            "fraction125to180um",
            "fraction180to250um",
            "fraction250to355um",
            "fraction355to500um",
            "fraction500to710um",
            "fraction710to1000um",
            "fraction1000to1400um",
            "fraction1400umto2mm",
            "fraction2to4mm",
            "fraction4to8mm",
            "fraction16to31_5mm",
            "fraction31_5to63mm",
            "fractionLarger63mm",
        ],
    },
}

# WIERTSEMA = {
#     "payload_root": "ns1:sourceDocument",
#     "nr": {"xpath": "ns1:objectIdAccountableParty"},
#     "location": {
#         "xpath": "ns1:deliveredLocation/ns3:location/ns2:Point/ns2:pos",
#         "resolver": resolvers.parse_coordinates,
#         "el-attr": "text",
#     },
#     "crs": {
#         "xpath": "ns1:deliveredLocation/ns3:location/ns2:Point",
#         "resolver": resolvers.parse_crs,
#     },
#     "surface": {
#         "xpath": "ns1:deliveredVerticalPosition/ns3:offset",
#         "resolver": resolvers.safe_float,
#         "el-attr": "text",
#     },
#     "vertical_datum": {
#         "xpath": "ns1:deliveredVerticalPosition/ns3:verticalDatum",
#         "el-attr": "text",
#     },
#     "groundwater_level": {
#         "xpath": "ns1:boring/ns3:groundwaterLevel",
#         "resolver": resolvers.safe_float,
#         "el-attr": "text",
#     },
#     "end": {
#         "xpath": "ns1:boring/ns3:finalDepthBoring",
#         "resolver": resolvers.safe_float,
#         "el-attr": "text",
#     },
#     "data": {
#         "xpath": "ns1:boreholeSampleDescription/ns3:descriptiveBoreholeLog",
#         "resolver": resolvers.process_bhrgt_data,
#         "layer-attributes": ["upperBoundary", "lowerBoundary", "geotechnicalSoilName"],
#     },
# }

schemas = {"BRO": BRO}  # , "Wiertsema": WIERTSEMA}
