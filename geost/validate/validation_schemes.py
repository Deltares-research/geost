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
        "depth": Column(numeric),
    }
    dataschema_inclined_point = {
        "x_bot": Column(numeric),
        "y_bot": Column(numeric),
    }
    dataschema_grainsize_data = {
        "sample_nr": Column(stringlike),
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "top": Column(numeric),
        "bottom": Column(numeric, checks=Check(">", "top", report_by="sample_nr")),
        "d_low": Column(numeric),
        "d_high": Column(numeric, checks=Check(">", "d_low", report_by="sample_nr")),
        "percentage": Column(numeric, checks=Check(">=", 0, report_by="sample_nr")),
    }

    # Below schemas are not yet in use
    boreholeschema = {
        "lith": Column(stringlike),
    }
    cptschema = {
        "length": Column(float),
        "qc": Column(float),
    }
