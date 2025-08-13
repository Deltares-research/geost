import warnings
from typing import NamedTuple

import pandas as pd
from geopandas.array import GeometryDtype
from pandera.errors import SchemaError, SchemaErrors
from pandera.pandas import Check, Column, DataFrameSchema

from geost import config
from geost._warnings import ValidationWarning


def safe_validate(schema: DataFrameSchema, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Catch validation errors and raise a warning instead while returning the DataFrame
    unchanged. If rows are dropped due to validation, report which rows were dropped.

    Parameters
    ----------
    schema : DataFrameSchema
        The schema to validate the DataFrame against.
    df : pd.DataFrame
        The DataFrame to validate.
    **kwargs : keyword arguments
        Additional arguments to pass to the Pandera DataFrameSchema.validate method. Note
        that the inplace argument is ignored and the DataFrame will not be modified in place.
        i.e. this function always returns a DataFrame.

    Returns
    -------
    pd.DataFrame
        Either the validated (and possibly changed) DataFrame or the unchanged DataFrame
        if validation failed.
    """
    if "inplace" in kwargs:
        raise ValueError("'inplace' argument is not supported.")

    # Update schema drop invalid rows setting, adjust lazy validation accordingly
    # (Pandera requires lazy validation when drop_invalid_rows is True)
    schema.drop_invalid_rows = config.validation.DROP_INVALID
    kwargs["lazy"] = True if config.validation.DROP_INVALID else False

    try:
        validated_df = schema.validate(df, **kwargs)
        if config.validation.DROP_INVALID:
            dropped = df.index.difference(validated_df.index)
            if len(dropped) > 0 and config.validation.VERBOSE:
                warnings.warn(
                    f"\nValidation dropped {len(dropped)} row(s) for schema '{schema.name}'.\n"
                    f"Dropped indices: {list(dropped)}\n",
                    category=ValidationWarning,
                )
        return validated_df
    except (SchemaError, SchemaErrors) as e:
        if config.validation.VERBOSE:
            warnings.warn(
                f"\nValidation failed for schema '{schema.name}'.\nDetails:\n{str(e)}\n",
                category=ValidationWarning,
            )
        return df


class DataSchemas(NamedTuple):
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
        checks=Check(lambda df: df["end"] < df["surface"], element_wise=False),
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
        checks=Check(lambda df: df["bottom"] >= df["top"], element_wise=False),
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
        checks=Check(lambda df: df["bottom"] >= df["top"], element_wise=False),
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
        checks=Check(lambda df: df["depth"] >= 0, element_wise=False),
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
        checks=Check(lambda df: df["depth"] >= 0, element_wise=False),
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
        drop_invalid_rows=config.validation.DROP_INVALID,
        name="Grain size data",
        description="Schema for validating grain size data",
    )
