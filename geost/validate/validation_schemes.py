from typing import NamedTuple

from geost.validate.validate import Check, Column, numeric, stringlike


class ValidationSchemas(NamedTuple):
    headerschema_point = {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "surface": Column(numeric),
        "end": Column(numeric, checks=Check("<", "surface", report_by="nr")),
    }
    dataschema_layered_point = {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "surface": Column(numeric),
        "end": Column(numeric),
        "top": Column(numeric),
        "bottom": Column(numeric, checks=Check(">", "top", report_by="nr")),
    }
    dataschema_discrete_point = {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "surface": Column(numeric),
        "end": Column(numeric),
        "z": Column(numeric),
    }
    dataschema_inclined_point = {
        "x_bot": Column(numeric),
        "y_bot": Column(numeric),
    }

    # Below schemas are not yet in use
    boreholeschema = {
        "lith": Column(stringlike),
    }
    cptschema = {
        "length": Column(float),
        "qc": Column(float),
    }
