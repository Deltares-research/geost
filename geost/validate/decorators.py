from functools import wraps

from geost.validate.validate import DataFrameSchema
from geost.validate.validation_schemes import ValidationSchemas

validationschemas = ValidationSchemas()


def validate_header(f):
    @wraps(f)
    def wrapper(*args):
        dataframe_to_validate = args[1]
        validation_instance = DataFrameSchema(
            "Header validation", validationschemas.headerschema_point
        )
        validation_instance.validate(dataframe_to_validate)
        return f(*args)

    return wrapper


def validate_data(f):
    @wraps(f)
    def wrapper(*args):
        data_object = args[0]
        dataframe_to_validate = args[1]
        dtype = data_object.datatype

        # Determine what the validation conditions are and append to 'schema_to_use'
        if dtype == "layered":
            schema_to_use = validationschemas.dataschema_layered_point
        elif dtype == "discrete":
            schema_to_use = validationschemas.dataschema_discrete_point

        if data_object.has_inclined:
            schema_to_use = schema_to_use | validationschemas.dataschema_inclined_point

        # Use the acquired validation conditions for validation
        validation_instance = DataFrameSchema("Data validation", schema_to_use)
        validation_instance.validate(dataframe_to_validate)

        return f(*args)

    return wrapper
