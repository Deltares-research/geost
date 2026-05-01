import warnings
from functools import reduce
from typing import NamedTuple

import geopandas as gpd
import pandas as pd

from geost import config
from geost._warnings import ValidationWarning


class ValidationResult:
    def __init__(self):
        self.errors = {}

    def __len__(self):
        # Note: length of unique validation issues, not number of affected rows or surveys
        return len(self.errors)

    def __repr__(self):
        return f"ValidationResult(num_issues={len(self)})"

    @property
    def has_errors(self) -> bool:
        """Indicates whether any validation errors were recorded."""
        return len(self.errors) > 0

    @property
    def error_indices(self) -> pd.Index:
        """Returns the indices of all rows with validation errors."""
        return reduce(
            lambda acc, error: acc.union(error["indices"]),
            self.errors.values(),
            pd.Index([]),
        ).unique()

    @property
    def error_nrs(self) -> list:
        """Returns the unique survey numbers associated with validation errors."""
        nrs = set()
        for error in self.errors.values():
            nrs.update(error["nrs"])
        return list(nrs)

    def add(
        self,
        column: str,
        error_message: str,
        affected_nrs: list,
        affected_indices: pd.Index,
    ):
        """
        Add a validation error to the result.

        Parameters
        ----------
        column : str
            The name of the column where the error occurred.
        error_message : str
            The error message describing the validation issue.
        affected_nrs : list
            The list of affected survey numbers.
        affected_indices : pd.Index
            The indices of the affected rows in the DataFrame.
        """
        if error_message is not None:
            self.errors[column] = {
                "warning": error_message,
                "nrs": affected_nrs,
                "indices": affected_indices,
            }

    def display_warnings(self):
        """Display all recorded validation warnings in a readable format."""
        if self.has_errors:
            for i, (column, error_data) in enumerate(self.errors.items()):
                warnings.warn(
                    f"\n{'=' * 60}\n"
                    f"⚠️  VALIDATION ISSUE ({i + 1}/{len(self)})\n"
                    f"{'=' * 60}\n"
                    f"Column    : '{column}'\n"
                    f"Message   : {error_data['warning']}\n"
                    f"# surveys : {len(error_data['nrs'])}\n"
                    f"# rows    : {len(error_data['indices'])}\n"
                    f"{'=' * 60}\n",
                    category=ValidationWarning,
                    source=None,
                )

    def handle_errors(self, df: pd.DataFrame, nr_col: str = None) -> pd.DataFrame:
        """
        Handle validation errors by flagging or dropping invalid rows depending on geost
        configuration settings.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to handle validation errors for.

        Returns
        -------
        pd.DataFrame
            The DataFrame with validation errors handled.
        """
        df_validated = df.copy()
        if self.has_errors:
            if config.validation.FLAG_INVALID:
                df_validated["is_valid"] = ~df_validated.index.isin(self.error_indices)
            if config.validation.DROP_INVALID:
                df_validated = df_validated[~df_validated[nr_col].isin(self.error_nrs)]

            if config.validation.VERBOSE:
                if (
                    config.validation.FLAG_INVALID
                    and not config.validation.DROP_INVALID
                ):
                    print(
                        f"\n{'\u2705'} Invalid rows were flagged with an 'is_valid' column because"
                        " geost.config.validation.FLAG_INVALID=True and geost.config.validation.DROP_INVALID=False"
                    )
                elif config.validation.DROP_INVALID:
                    print(
                        f"\n{'\u2705'} Invalid surveys were dropped from the DataFrame because"
                        " geost.config.validation.DROP_INVALID=True"
                    )
                else:
                    print(
                        f"\n{'\u274c'} Invalid surveys were retained in the DataFrame because geost.config.validation.FLAG_INVALID and DROP_INVALID are False"
                    )

                print(
                    f"\n{'\U0001f4d6'} See the user guide section on validation for advanced handling of validation issues: https://deltares-research.github.io/geost/user_guide/validation.html"
                )
        return df_validated


def coerce_numeric(
    obj: pd.DataFrame,
    columns: str | list[str],
    validation_result: ValidationResult,
    nullable: bool = False,
) -> pd.DataFrame:
    """
    Attempt to coerce a column or columns to numeric, raising a ValueError if coercion
    fails.

    Parameters
    ----------
    obj : pd.DataFrame
        The DataFrame containing the column to coerce.
    column : str
        The name(s) of the column(s) to coerce.
    validation_result : ValidationResult
        The ValidationResult object to record any validation errors.
    nullable : bool, optional
        Whether to allow null values in the column. If False, any null values will cause
        coercion to fail. Default is False.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified column coerced to numeric.

    Raises
    ------

    ValueError
        If the column cannot be coerced to numeric.

    """
    for column in columns:
        if not pd.api.types.is_numeric_dtype(obj[column]):
            obj[column] = pd.to_numeric(obj[column], errors="coerce")
            if not nullable:
                error_indices = obj[obj[column].isna()].index
                error_nrs = obj["nr"][obj[column].isna()].unique().tolist()
                if len(error_nrs) > 0:
                    warning = f"Column '{column}' must contain only numeric values, but some values could not be coerced. "
                    validation_result.add(column, warning, error_nrs, error_indices)

        elif pd.api.types.is_numeric_dtype(obj[column]) and not nullable:
            error_indices = obj[obj[column].isna()].index
            error_nrs = obj["nr"][obj[column].isna()].unique().tolist()
            if len(error_nrs) > 0:
                warning = f"Column '{column}' must not contain NaN values, but some values are NaN."
                validation_result.add(column, warning, error_nrs, error_indices)

    return obj


def validate_base(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    column_names: dict,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Perform basic validation checks on the input DataFrame or GeoDataFrame.
    Checks for required columns and attempts to coerce the 'surface' column to numeric.

    Parameters
    ----------
    obj : pd.DataFrame | gpd.GeoDataFrame
        The DataFrame or GeoDataFrame to validate.
    column_names : dict
        A dictionary containing the names of GeostFrame positional columns.

    Raises
    ------
    ValueError
        If required columns are missing.

    """
    if column_names["nr"] not in obj.columns:
        raise ValueError(f"GeostFrame missing required column: '{column_names['nr']}'")
    if column_names["surface"] not in obj.columns:
        raise ValueError(
            f"GeostFrame missing required column: '{column_names['surface']}'"
        )


def validate_top_bottom(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    column_names: dict,
    first_row_in_survey: pd.Series,
    validation_result: ValidationResult,
):
    # Ensure top is above bottom (i.e. top < bottom)
    invalid_indices = obj[obj[column_names["top"]] >= obj[column_names["depth"]]].index
    invalid_nrs = obj["nr"].loc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Column '{column_names['top']}' must be less than '{column_names['depth']}', but some rows violate this condition."
    else:
        warning = None

    validation_result.add(
        f"{column_names['top']}, {column_names['depth']}",
        warning,
        invalid_nrs,
        invalid_indices,
    )

    # Ensure top and bottom are positive downwards (i.e. >= 0)
    invalid_indices = obj[
        (
            (obj[column_names["top"]].diff() <= 0.0)
            | (obj[column_names["depth"]].diff() <= 0.0)
        )
        & (~first_row_in_survey)
    ].index
    invalid_nrs = obj["nr"].iloc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Columns '{column_names['top']}' and '{column_names['depth']}' must be positive downwards (i.e. >= 0), but some rows violate this condition."
    else:
        warning = None

    validation_result.add(
        f"{column_names['top']}, {column_names['depth']}",
        warning,
        invalid_nrs,
        invalid_indices,
    )


def validate_depths(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    column_names: dict,
    first_row_in_survey: pd.Series,
    validation_result: ValidationResult,
):
    # Ensure depth column is positive downwards (i.e. >= 0)
    invalid_indices = obj[
        (obj[column_names["depth"]].diff() < 0) & (~first_row_in_survey)
    ].index
    invalid_nrs = obj["nr"].iloc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Column '{column_names['depth']}' must be positive downwards (i.e. >= 0), but some rows violate this condition."
    else:
        warning = None

    validation_result.add(
        f"{column_names['depth']}", warning, invalid_nrs, invalid_indices
    )


def validate_geostframe(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    has_depth_columns: bool = None,
    is_layered: bool = None,
    first_row_in_survey: pd.Series = None,
    positional_columns: dict = None,
):
    """
    Validate a GeostFrame, display warnings and handle errors as specified by the GeoST
    configuration.

    Parameters
    ----------
    obj : pd.DataFrame | gpd.GeoDataFrame
        The GeostFrame to be validated.
    has_depth_columns : bool, optional
        Indicates if the GeostFrame has depth columns, by default None
    is_layered : bool, optional
        Indicates if the GeostFrame is layered, by default None
    first_row_in_survey : pd.Series, optional
        Indicates the first row in each survey, by default None
    positional_columns : dict, optional
        A dictionary containing the names of GeostFrame positional columns, by default None

    Returns
    -------
    ValidationResult
        The result of the validation, a :class:`~geost.validation.validate.ValidationResult`
        object.
    """
    validation_result = ValidationResult()

    numeric_columns = [
        col_name
        for col_name in (
            positional_columns["surface"],
            positional_columns["x"],
            positional_columns["y"],
            positional_columns["top"],
            positional_columns["depth"],
        )
        if col_name is not None
    ]

    validate_base(obj, positional_columns)
    validated_obj = coerce_numeric(obj, numeric_columns, validation_result)

    if has_depth_columns:
        if is_layered:
            validate_top_bottom(
                validated_obj,
                positional_columns,
                first_row_in_survey,
                validation_result,
            )
        else:
            validate_depths(
                validated_obj,
                positional_columns,
                first_row_in_survey,
                validation_result,
            )

    validation_result.display_warnings()
    validated_obj = validation_result.handle_errors(
        validated_obj, positional_columns["nr"]
    )

    return validated_obj, validation_result
