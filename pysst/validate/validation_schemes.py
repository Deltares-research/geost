from pysst.validate import Check, Column, DataFrameSchema, numeric, stringlike

headerschema = DataFrameSchema(
    "header validation",
    {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "mv": Column(numeric),
        "end": Column(numeric, checks=Check("<", "mv", report_by="nr")),
    },
)

common_dataschema = DataFrameSchema(
    "layer data validation",
    {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "mv": Column(numeric),
        "end": Column(numeric),
        "top": Column(numeric),
        "bottom": Column(numeric, checks=Check("<", "top", report_by="nr")),
    },
)

inclined_dataschema = DataFrameSchema(
    "inclined layer data additional validation",
    {
        "x_bot": Column(numeric),
        "y_bot": Column(numeric),
    },
)

common_dataschema_depth_reference = DataFrameSchema(
    "layer data (with vertical reference=depth) validation",
    {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "mv": Column(numeric),
        "end": Column(numeric),
        "top": Column(numeric),
        "bottom": Column(numeric, checks=Check(">", "top", report_by="nr")),
    },
)

boreholeschema = DataFrameSchema(
    "borehole-specific validation",
    {
        "lith": Column(stringlike),
    },
)

cptschema = DataFrameSchema(
    "CPT-specific validation",
    {
        "length": Column(float),
        "qc": Column(float),
    },
)
