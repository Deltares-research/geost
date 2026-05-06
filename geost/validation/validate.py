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
        if not self.has_errors:
            return df

        df_validated = df.copy()
        if config.validation.FLAG_INVALID:
            df_validated["is_valid"] = ~df_validated.index.isin(self.error_indices)

        if config.validation.DROP_INVALID:
            df_validated = df_validated[~df_validated[nr_col].isin(self.error_nrs)]

        if config.validation.VERBOSE:
            if config.validation.FLAG_INVALID and not config.validation.DROP_INVALID:
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
                isna = obj[column].isna()
                if isna.any():
                    errors = obj[isna]
                    error_nrs = errors[obj.gst._nr].unique().tolist()
                    warning = f"Column '{column}' must contain only numeric values, but some values could not be coerced. "
                    validation_result.add(column, warning, error_nrs, errors.index)

        elif pd.api.types.is_numeric_dtype(obj[column]) and not nullable:
            isna = obj[column].isna()
            if isna.any():
                errors = obj[isna]
                error_nrs = errors[obj.gst._nr].unique().tolist()
                warning = f"Column '{column}' must not contain NaN values, but some values are NaN."
                validation_result.add(column, warning, error_nrs, errors.index)

    return obj, validation_result


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
    nr_, top_, depth_ = column_names["nr"], column_names["top"], column_names["depth"]

    # Ensure top is above bottom (i.e. top < bottom)
    invalid_top_above_bottom = obj[top_] >= obj[depth_]
    if invalid_top_above_bottom.any():
        invalid = obj[invalid_top_above_bottom]
        invalid_nrs = invalid[nr_].unique().tolist()
        warning = f"Column '{top_}' must be less than '{depth_}', but some rows violate this condition."

        validation_result.add(
            f"{top_}, {depth_}",
            warning,
            invalid_nrs,
            invalid.index,
        )

    # Ensure top and bottom are positive downwards (i.e. >= 0)
    invalid_depth_direction = (
        (obj[top_].diff() <= 0.0) | (obj[depth_].diff() <= 0.0)
    ) & (~first_row_in_survey)

    if invalid_depth_direction.any():
        invalid = obj[invalid_depth_direction]
        invalid_nrs = invalid[nr_].unique().tolist()
        warning = f"Columns '{top_}' and '{depth_}' must be positive downwards (i.e. >= 0), but some rows violate this condition."

        validation_result.add(
            f"{top_}, {depth_}",
            warning,
            invalid_nrs,
            invalid.index,
        )
    return validation_result


def validate_depths(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    column_names: dict,
    first_row_in_survey: pd.Series,
    validation_result: ValidationResult,
):
    nr_, depth_ = column_names["nr"], column_names["depth"]

    # Ensure depth column is positive downwards (i.e. >= 0)
    invalid_depth_direction = (obj[depth_].diff() <= 0.0) & (~first_row_in_survey)
    if invalid_depth_direction.any():
        invalid = obj[invalid_depth_direction]
        invalid_nrs = invalid[nr_].unique().tolist()
        warning = f"Column '{depth_}' must be positive downwards (i.e. >= 0), but some rows violate this condition."

        validation_result.add(
            f"{depth_}",
            warning,
            invalid_nrs,
            invalid.index,
        )
    return validation_result


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
    validated_obj : pd.DataFrame | gpd.GeoDataFrame
        The validated GeostFrame with errors handled according to configuration settings.
    validation_result : ValidationResult
        The result of the validation, a :class:`~geost.validation.validate.ValidationResult`
        object.

    """
    validation_result = ValidationResult()

    numeric_columns = [
        col
        for name in ["surface", "x", "y", "top", "depth"]
        if (col := positional_columns[name]) is not None
    ]

    validate_base(obj, positional_columns)
    validated_obj, validation_result = coerce_numeric(
        obj, numeric_columns, validation_result
    )

    if has_depth_columns:
        if is_layered:
            validation_result = validate_top_bottom(
                validated_obj,
                positional_columns,
                first_row_in_survey,
                validation_result,
            )
        else:
            validation_result = validate_depths(
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
