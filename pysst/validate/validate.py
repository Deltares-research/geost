import operator
from typing import Union, Optional
import warnings


OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}

raise_error = False


class ValidationError(Exception):
    pass


class ValidationWarning(Warning):
    pass


class Numeric:
    associated_types: tuple = (int, float, "int64", "float64")

    def __repr__(self):
        return "numeric type"

    def __eq__(self, other):
        return other in self.associated_types


numeric = Numeric()


class StringLike:
    associated_types: tuple = (str, "object")

    def __repr__(self):
        return "stringlike type"

    def __eq__(self, other):
        return other in self.associated_types


stringlike = StringLike()


class DataFrameSchema:
    def __init__(self, name: str, json_schema):
        self.name = name
        self.schema = json_schema
        self.validationerrors = []
        self.column_name = ""
        self.column_validation_parameters = None
        self.dataframe = None

    def _validate_dtype(self):
        column_dtype = self.dataframe[self.column_name].dtype
        if not self.column_validation_parameters.required_dtype == column_dtype:
            self.validationerrors.append(
                f'During {self.name}: datatype in column "{self.column_name}" is "{column_dtype}", but is required to be "{self.column_validation_parameters.required_dtype}"'
            )
            return False
        return True

    def _validate_checks(self):
        for check in self.column_validation_parameters.checks:
            success, reports = check.check(self.dataframe, self.column_name)
            if not success:
                self.validationerrors.append(
                    f'During {self.name}: data in column "{self.column_name}" failed check "{check}" for {len(reports)} rows: {reports}'
                )
                return False
            return True

    def validate(self, dataframe):
        self.dataframe = dataframe
        for self.column_name, self.column_validation_parameters in self.schema.items():
            if not self.column_name in dataframe.columns:
                self.validationerrors.append(
                    f'During {self.name}: required column "{self.column_name}" is missing'
                )
            else:
                passed_dtype_check = self._validate_dtype()
                if passed_dtype_check and self.column_validation_parameters.checks:
                    passed_custom_checks = self._validate_checks()

        if len(self.validationerrors) >= 1:
            if raise_error:
                raise ValidationError(("\n").join(self.validationerrors))
            else:
                print(
                    ValidationWarning(
                        "WARNING:\n--------\n"
                        + ("\n").join(self.validationerrors)
                        + "\n--------\n>> CONTINUING MAY LEAD TO UNEXPECTED RESULTS\n"
                    )
                )


class Check:
    def __init__(self, operator: str, reference_value, report_by="index"):
        self.operator_str = operator
        self.operator = OPERATORS[operator]
        self.reference = reference_value
        self.report_by = report_by

    def __repr__(self):
        return f"{self.operator_str} {self.reference}"

    def check(self, dataframe, column):
        if self.reference in dataframe.columns:
            reference = dataframe[self.reference]
        else:
            reference = self.reference

        comparison = self.operator(dataframe[column], reference)
        if not all(comparison):
            invalid_indices = dataframe[~comparison].index
            if self.report_by in dataframe.columns:
                return False, dataframe.loc[invalid_indices, self.report_by].values
            else:
                return False, invalid_indices
        else:
            return True, []


class Column:
    def __init__(self, required_dtype, checks=None):
        self.required_dtype = required_dtype
        if checks is not None:
            if checks.__class__ == Check:
                self.checks = [checks]
            elif len(checks) > 1:
                self.checks = checks
        else:
            self.checks = None

    def __repr__(self):
        return f"Column of type {self.required_dtype}"
