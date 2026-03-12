import warnings
from functools import reduce

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
                )

    def handle_errors(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if self.has_errors:
            if config.validation.FLAG_INVALID:
                df["is_valid"] = ~df.index.isin(self.error_indices)
            if config.validation.DROP_INVALID:
                df.drop(index=self.error_indices, inplace=True)

        if config.validation.VERBOSE:
            if config.validation.FLAG_INVALID and not config.validation.DROP_INVALID:
                print(
                    f"\n{'\u2705'} Invalid rows were flagged with an 'is_valid' column because"
                    " geost.config.validation.FLAG_INVALID=True and geost.config.validation.DROP_INVALID=False"
                )
            elif config.validation.DROP_INVALID:
                print(
                    f"\n{'\u2705'} Invalid rows were dropped from the DataFrame because"
                    " geost.config.validation.DROP_INVALID=True"
                )
            else:
                print(
                    f"\n{'\u274c'} Invalid rows were retained in the DataFrame because geost.config.validation.FLAG_INVALID and DROP_INVALID are False"
                )

            print(
                f"\n{'\U0001f4d6'} See the user guide section on validation for advanced handling of validation issues: https://deltares-research.github.io/geost/user_guide/validation.html"
            )


def coerce_numeric(
    obj: pd.DataFrame, column: str, nullable: bool = False
) -> pd.DataFrame:
    """
    Attempt to coerce a column to numeric, raising a ValueError if coercion fails.

    Parameters
    ----------
    obj : pd.DataFrame
        The DataFrame containing the column to coerce.
    column : str
        The name of the column to coerce.
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
    if not pd.api.types.is_numeric_dtype(obj[column]):
        obj[column] = pd.to_numeric(obj[column], errors="coerce")
        error_indices = obj[obj[column].isna()].index
        error_nrs = obj["nr"][obj[column].isna()].unique().tolist()
        if len(error_nrs) > 0:
            warning = f"Column '{column}' must contain only numeric values, but some values could not be coerced. "
        else:
            warning = None
    else:
        error_indices = pd.Index([])
        error_nrs = []
        warning = None

    return obj, error_nrs, error_indices, warning


def validate_base(
    obj: pd.DataFrame | gpd.GeoDataFrame, validation_result: ValidationResult
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Perform basic validation checks on the input DataFrame or GeoDataFrame.
    Checks for required columns and attempts to coerce the 'surface' column to numeric.

    Parameters
    ----------
    obj : pd.DataFrame | gpd.GeoDataFrame
        The DataFrame or GeoDataFrame to validate.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        The validated DataFrame or GeoDataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.

    """
    if "nr" not in obj.columns:
        raise ValueError("GeostFrame missing required column: 'nr'")
    if "surface" not in obj.columns:
        raise ValueError("GeostFrame missing required column: 'surface'")

    obj, error_nrs, error_indices, warning = coerce_numeric(obj, "surface")
    validation_result.add("surface", warning, error_nrs, error_indices)
    return obj


def validate_xy(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    x_col: str,
    y_col: str,
    validation_result: ValidationResult,
):
    obj, x_errors, x_error_indices, x_warning = coerce_numeric(obj, x_col)
    obj, y_errors, y_error_indices, y_warning = coerce_numeric(obj, y_col)

    validation_result.add(x_col, x_warning, x_errors, x_error_indices)
    validation_result.add(y_col, y_warning, y_errors, y_error_indices)

    return obj


def validate_top_bottom(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    top_col: str,
    bottom_col: str,
    first_row_in_survey: pd.Series,
    validation_result: ValidationResult,
):
    # Coerce top and bottom to numeric and collect any coercion errors
    obj, top_errors, top_error_indices, top_warning = coerce_numeric(obj, top_col)
    obj, bottom_errors, bottom_error_indices, bottom_warning = coerce_numeric(
        obj, bottom_col
    )

    validation_result.add(top_col, top_warning, top_errors, top_error_indices)
    validation_result.add(
        bottom_col, bottom_warning, bottom_errors, bottom_error_indices
    )

    # Ensure top is above bottom (i.e. top < bottom)
    invalid_indices = obj[obj[top_col] >= obj[bottom_col]].index
    invalid_nrs = obj["nr"].loc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Column '{top_col}' must be less than '{bottom_col}', but some rows violate this condition."
    else:
        warning = None

    validation_result.add(
        f"{top_col}, {bottom_col}", warning, invalid_nrs, invalid_indices
    )

    # Ensure first top is 0
    invalid_indices = obj[first_row_in_survey & (obj[top_col] != 0.0)].index
    invalid_nrs = obj["nr"].loc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Column '{top_col}' must be 0 for the first row in each survey, but some rows violate this condition."
    else:
        warning = None

    validation_result.add(f"{top_col}", warning, invalid_nrs, invalid_indices)

    # Ensure top and bottom are positive downwards (i.e. >= 0)
    invalid_indices = obj[
        ((obj[top_col].diff() <= 0.0) | (obj[bottom_col].diff() <= 0.0))
        & (~first_row_in_survey)
    ].index
    invalid_nrs = obj["nr"].iloc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Columns '{top_col}' and '{bottom_col}' must be positive downwards (i.e. >= 0), but some rows violate this condition."
    else:
        warning = None

    validation_result.add(
        f"{top_col}, {bottom_col}", warning, invalid_nrs, invalid_indices
    )

    return obj


def validate_depths(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    bottom_col: str,
    first_row_in_survey: pd.Series,
    validation_result: ValidationResult,
):
    obj, bottom_errors, bottom_error_indices, bottom_warning = coerce_numeric(
        obj, bottom_col
    )
    validation_result.add(
        bottom_col, bottom_warning, bottom_errors, bottom_error_indices
    )

    # Ensure first top is 0
    if first_row_in_survey is not None:
        invalid_indices = obj[first_row_in_survey & (obj[bottom_col] != 0.0)].index
        invalid_nrs = obj["nr"].loc[invalid_indices].unique().tolist()
        if len(invalid_indices) > 0:
            warning = f"Column '{bottom_col}' must be 0 for the first row in each survey, but some rows violate this condition."
        else:
            warning = None

        validation_result.add(f"{bottom_col}", warning, invalid_nrs, invalid_indices)

    # Ensure depth column is positive downwards (i.e. >= 0)
    invalid_indices = obj[(obj[bottom_col].diff() < 0) & (~first_row_in_survey)].index
    invalid_nrs = obj["nr"].iloc[invalid_indices].unique().tolist()
    if len(invalid_indices) > 0:
        warning = f"Column '{bottom_col}' must be positive downwards (i.e. >= 0), but some rows violate this condition."
    else:
        warning = None

    validation_result.add(f"{bottom_col}", warning, invalid_nrs, invalid_indices)

    return obj


def validate_geostframe(
    obj: pd.DataFrame | gpd.GeoDataFrame,
    has_depth_columns: bool = None,
    is_layered: bool = None,
    has_xy_columns: bool = None,
    x_col: str = None,
    y_col: str = None,
    top_col: str = None,
    bottom_col: str = None,
    first_row_in_survey: pd.Series = None,
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
    has_xy_columns : bool, optional
        Indicates if the GeostFrame has XY columns, by default None
    x_col : str, optional
        Name of the X column, by default None
    y_col : str, optional
        Name of the Y column, by default None
    top_col : str, optional
        Name of the top column, by default None
    bottom_col : str, optional
        Name of the bottom column, by default None
    first_row_in_survey : pd.Series, optional
        Indicates the first row in each survey, by default None

    Returns
    -------
    ValidationResult
        The result of the validation, a :class:`~geost.validation.validate.ValidationResult`
        object.
    """
    validation_result = ValidationResult()
    validated_obj = validate_base(obj, validation_result)

    if has_depth_columns:
        if is_layered:
            validated_obj = validate_top_bottom(
                validated_obj,
                top_col,
                bottom_col,
                first_row_in_survey,
                validation_result,
            )
        else:
            validated_obj = validate_depths(
                validated_obj, bottom_col, first_row_in_survey, validation_result
            )
    if has_xy_columns:
        validated_obj = validate_xy(validated_obj, x_col, y_col, validation_result)

    validation_result.display_warnings()
    validation_result.handle_errors(validated_obj)

    return validation_result
