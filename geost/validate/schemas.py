from typing import NamedTuple

from pandera.pandas import Check, Column, DataFrameSchema


class ValidationSchemas(NamedTuple):
    # Layer data schema
    layerdata = DataFrameSchema(
        columns={
            "nr": Column(str),
            "x": Column(float),
            "y": Column(float),
            "surface": Column(float, nullable=True),
            "end": Column(float, nullable=True),
            "top": Column(float),
            "bottom": Column(float),
        },
        checks=[
            Check(lambda df: df["bottom"] >= df["top"], element_wise=False),
        ],
        coerce=True,
        strict=False,
        title="Layer data",
        description="Schema for validating generic layer data",
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
            "d_high": Column(float, nullable=True),
            "percentage": Column(float),
        },
        checks=[
            Check(lambda df: df["bottom"] >= df["top"], element_wise=False),
            Check(lambda df: df["d_high"] > df["d_low"], element_wise=False),
            Check(
                lambda df: (df["percentage"] >= 0) & (df["percentage"] <= 100),
                element_wise=False,
            ),
        ],
        coerce=True,
        strict=False,
        title="Grain size data",
        description="Schema for validating grain size data",
    )
