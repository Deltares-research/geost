import operator
from typing import Union, Optional

OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


class ValidationError(Exception):
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

    def validate(self, dataframe):
        for column_name, column_validation_parameters in self.schema.items():
            column_dtype = dataframe[column_name].dtype
            if not column_validation_parameters.required_dtype == column_dtype:
                self.validationerrors.append(
                    f'During {self.name}: datatype in column "{column_name}" is "{column_dtype}", but is required to be "{column_validation_parameters.required_dtype}"'
                )
                dtype_ok = False
            else:
                dtype_ok = True

            if column_validation_parameters.checks and dtype_ok:
                for check in column_validation_parameters.checks:
                    success, reports = check.check(dataframe, column_name)
                    if not success:
                        self.validationerrors.append(
                            f'During {self.name}: data in column "{column_name}" failed check "{check}" for {len(reports)} rows: {reports}'
                        )

        if len(self.validationerrors) >= 1:
            raise ValidationError(self.validationerrors)


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
