from typing import NamedTuple

from geost.validate.validate import Check, Column, numeric, stringlike


class ValidationSchemas(NamedTuple):
    headerschema = {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "surface": Column(numeric),
        "end": Column(numeric, checks=Check("<", "surface", report_by="nr")),
    }
    common_dataschema = {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "surface": Column(numeric),
        "end": Column(numeric),
        "top": Column(numeric),
        "bottom": Column(numeric, checks=Check("<", "top", report_by="nr")),
    }
    inclined_dataschema = {
        "x_bot": Column(numeric),
        "y_bot": Column(numeric),
    }
    common_dataschema_depth_reference = {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "surface": Column(numeric),
        "end": Column(numeric),
        "top": Column(numeric),
        "bottom": Column(numeric, checks=Check(">", "top", report_by="nr")),
    }
    boreholeschema = {
        "lith": Column(stringlike),
    }
    cptschema = {
        "length": Column(float),
        "qc": Column(float),
    }
