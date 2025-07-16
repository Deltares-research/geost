from geost.io.xml import resolvers

BRO = {
    "payload_root": "dispatchDocument",
    "nr": {"xpath": "brocom:broId"},
    "location": {
        "xpath": "./deliveredLocation/bhrgtcom:location/gml:Point/gml:pos",
        "resolver": resolvers.parse_coordinates,
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
}

SCHEMA = {"BRO": BRO, "Wiertsema": WIERTSEMA}
