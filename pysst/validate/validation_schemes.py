from pysst.validate import DataFrameSchema, Column, Check, numeric, stringlike

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
    "data layer data validation",
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

common_dataschema_depth_reference = DataFrameSchema(
    "borehole layer data (vertical reference=depth) validation",
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
    "borehole layer data validation",
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

boreholeschema_depth_reference = DataFrameSchema(
    "borehole layer data (vertical reference=depth) validation",
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

cptschema = DataFrameSchema(
    "CPT layer data validation",
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

cptschema_depth_reference = DataFrameSchema(
    "CPT layer data (vertical reference=depth) validation",
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
