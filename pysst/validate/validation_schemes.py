from pysst.validate import DataFrameSchema, Column, Check, numeric, stringlike

headerschema = DataFrameSchema(
    "Header validation",
    {
        "nr": Column(stringlike),
        "x": Column(numeric),
        "y": Column(numeric),
        "mv": Column(numeric),
        "end": Column(numeric, checks=Check("<", "mv", report_by="nr")),
    },
)

boreholeschema = DataFrameSchema("Borehole layer data validation", {})
boreholeschema_depth_reference = DataFrameSchema(
    "Borehole layer data (vertical reference=depth) validation", {}
)
cptschema = DataFrameSchema("CPT layer data validation", {})
cptschema_depth_reference = DataFrameSchema(
    "CPT layer data (vertical reference=depth) validation", {}
)
