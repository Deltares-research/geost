import warnings

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from pandera.pandas import DataFrameSchema

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
        if config.validation.FLAG_INVALID:
            df["is_valid"] = ~df.index.isin(e.failure_cases["index"])
        return df
