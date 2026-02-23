from geopandas.array import GeometryDtype
from pandera.pandas import Check, Column, DataFrameSchema

from geost import config


def combine_schemas(*schemas: DataFrameSchema | None) -> DataFrameSchema:
    """
    Combine multiple DataFrameSchema objects into a single schema.

    Parameters:
        *schemas: Variable number of DataFrameSchema objects to combine. Can also be None,
        which will be ignored.

    Returns:
        A new DataFrameSchema that combines the columns and checks of all input schemas.
    """
    combined_columns = {}
    combined_checks = []
    for schema in schemas:
        if schema is not None:
            combined_columns.update(schema.columns)
            combined_checks.extend(schema.checks)
    return DataFrameSchema(
        columns=combined_columns,
        checks=combined_checks,
        coerce=True,
        strict=False,
        drop_invalid_rows=config.validation.DROP_INVALID,
        name="Combined schema",
        description="A schema that combines multiple schemas",
    )


geostframe_base = DataFrameSchema(
    columns={
        "nr": Column(str, nullable=False),
        "surface": Column(float, nullable=False),
    },
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="GeostFrame most basic type schema",
    description="Schema for validating GeostFrame that represents the most basic type of data with only nr and surface columns",
)

geostframe_with_top_bottom = DataFrameSchema(
    columns={
        "top": Column(float, nullable=False),
        "bottom": Column(float, nullable=False),
    },
    checks=[
        Check(
            lambda df: df["bottom"] >= df["top"] and df["top"] >= 0,
            element_wise=False,
            error="Bottom depth must be greater than or equal to top depth and top depth must be non-negative",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="GeostFrame data type schema with top/bottom depth data",
    description="Schema for validating GeostFrame that represents depth data with top and bottom columns",
)

geostframe_with_bottom = DataFrameSchema(
    columns={
        "nr": Column(str, nullable=False),
        "surface": Column(float, nullable=False, required=False),
        "bottom": Column(float, nullable=False),
    },
    checks=[
        Check(
            lambda df: df["bottom"] >= 0,
            element_wise=False,
            error="Bottom depth must be greater than 0",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="GeostFrame data type schema with only bottom depth data",
    description="Schema for validating GeostFrame that represents depth data with only bottom column",
)

geostframe_with_depth = DataFrameSchema(
    columns={
        "nr": Column(str, nullable=False),
        "surface": Column(float, nullable=False, required=False),
        "depth": Column(float, nullable=False),
    },
    checks=[
        Check(
            lambda df: df["depth"] >= 0,
            element_wise=False,
            error="Depth must be greater than 0",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="GeostFrame data type schema with only depth data",
    description="Schema for validating GeostFrame that represents depth data with only depth column",
)

geostframe_with_geometry = DataFrameSchema(
    columns={
        "geometry": Column(GeometryDtype(), nullable=False),
    },
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="GeostFrame with geometry",
    description="Addition for GeoSTFrame schemas that includes geometry column",
)

geostframe_with_xy = DataFrameSchema(
    columns={
        "x": Column(float, nullable=False),
        "y": Column(float, nullable=False),
    },
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="GeostFrame with XY coordinates",
    description="Addition for GeoSTFrame schemas that includes x and y coordinates",
)

# Point header schema
pointheader = DataFrameSchema(
    columns={
        "nr": Column(str, unique=True),
        "x": Column(float),
        "y": Column(float),
        "surface": Column(float),
        "end": Column(float),
        "geometry": Column(GeometryDtype()),
    },
    checks=Check(
        lambda df: df["end"] < df["surface"],
        element_wise=False,
        error="End depth must be lower than surface depth",
    ),
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Point header",
    description="Schema for validating point header data",
)

# Line header schema
lineheader = DataFrameSchema(
    columns={
        "nr": Column(str, unique=True),
        "geometry": Column(GeometryDtype(), nullable=False),
    },
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Line header",
    description="Schema for validating line header data",
)

# Layer data schema non-inclined
layerdata = DataFrameSchema(
    columns={
        "nr": Column(str),
        "x": Column(float),
        "y": Column(float),
        "surface": Column(float),
        "end": Column(float),
        "top": Column(float),
        "bottom": Column(float),
    },
    checks=[
        Check(
            lambda df: df["bottom"] >= df["top"],
            element_wise=False,
            error="Bottom depth must be greater than or equal to top depth",
        ),
        Check(
            lambda df: df["end"] < df["surface"],
            element_wise=False,
            error="End depth must be lower than surface depth",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Layer data non-inclined",
    description="Schema for validating generic layer data that is not inclined",
)

# Layer data schema inclined
layerdata_inclined = DataFrameSchema(
    columns={
        "nr": Column(str),
        "x": Column(float),
        "y": Column(float),
        "x_bot": Column(float),
        "y_bot": Column(float),
        "surface": Column(float),
        "end": Column(float),
        "top": Column(float),
        "bottom": Column(float),
    },
    checks=[
        Check(
            lambda df: df["bottom"] >= df["top"],
            element_wise=False,
            error="Bottom depth must be greater than or equal to top depth",
        ),
        Check(
            lambda df: df["end"] < df["surface"],
            element_wise=False,
            error="End depth must be lower than surface depth",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Layer data inclined",
    description="Schema for validating generic layer data that is inclined",
)

# Discrete data schema non-inclined
discretedata = DataFrameSchema(
    columns={
        "nr": Column(str),
        "x": Column(float),
        "y": Column(float),
        "surface": Column(float),
        "end": Column(float),
        "depth": Column(float),
    },
    checks=[
        Check(
            lambda df: df["depth"] >= 0,
            element_wise=False,
            error="Depth must be non-negative",
        ),
        Check(
            lambda df: df["end"] < df["surface"],
            element_wise=False,
            error="End depth must be lower than surface depth",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Discrete data non-inclined",
    description="Schema for validating generic discrete data that is not inclined",
)

# Discrete data schema inclined
discretedata_inclined = DataFrameSchema(
    columns={
        "nr": Column(str),
        "x": Column(float),
        "y": Column(float),
        "x_bot": Column(float),
        "y_bot": Column(float),
        "surface": Column(float),
        "end": Column(float),
        "depth": Column(float),
    },
    checks=[
        Check(
            lambda df: df["depth"] >= 0,
            element_wise=False,
            error="Depth must be non-negative",
        ),
        Check(
            lambda df: df["end"] < df["surface"],
            element_wise=False,
            error="End depth must be lower than surface depth",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Discrete data inclined",
    description="Schema for validating generic discrete data that is inclined",
)

# Grain size data schema
grainsize_data = DataFrameSchema(
    columns={
        "nr": Column(str),
        "sample_nr": Column(str),
        "x": Column(float),
        "y": Column(float),
        "top": Column(float),
        "bottom": Column(float),
        "d_low": Column(float),
        "d_high": Column(float),
        "percentage": Column(float, nullable=True, required=False),
        "mass": Column(float, nullable=True, required=False),
    },
    checks=[
        Check(
            lambda df: df["bottom"] >= df["top"],
            element_wise=False,
            error="Bottom depth must be greater than or equal to top depth",
        ),
        Check(
            lambda df: df["d_high"] > df["d_low"],
            element_wise=False,
            error="High diameter must be greater than low diameter",
        ),
        Check(
            lambda df: (df["percentage"] >= 0) & (df["percentage"] <= 100),
            element_wise=False,
            error="Percentage must be between 0 and 100",
        ),
        Check(
            lambda df: df["mass"] >= 0,
            element_wise=False,
            error="Mass must be greater than or equal to 0",
        ),
        Check(
            lambda df: (
                (df["percentage"].notna() & df["mass"].isna())
                | (df["percentage"].isna() & df["mass"].notna())
            ),
            element_wise=False,
            error="One of 'percentage' or 'mass' must be present (not both or neither).",
        ),
    ],
    coerce=True,
    strict=False,
    drop_invalid_rows=config.validation.DROP_INVALID,
    name="Grain size data",
    description="Schema for validating grain size data",
)
