from functools import wraps

from geost.validate.validate import DataFrameSchema
from geost.validate.validation_schemes import ValidationSchemas

validationschemas = ValidationSchemas()


def validate_header(f):
    @wraps(f)
    def wrapper(*args):
        dataframe_to_validate = args[1]
        validation_instance = DataFrameSchema(
            "Header validation", validationschemas.headerschema
        )
        validation_instance.validate(dataframe_to_validate)
        return f(*args)

    return wrapper


def validate_data(f):
    @wraps(f)
    def wrapper(*args):
        collection_object = args[0]
        dataframe_to_validate = args[1]

        # Determine what the validation conditions are and append to 'schema_to_use'
        if collection_object.vertical_reference == "depth":
            schema_to_use = validationschemas.common_dataschema_depth_reference
        else:
            schema_to_use = validationschemas.common_dataschema

        if collection_object.is_inclined:
            schema_to_use = schema_to_use | validationschemas.inclined_dataschema

        # Use the acquired validation conditions for validation
        validation_instance = DataFrameSchema("Data validation", schema_to_use)
        validation_instance.validate(dataframe_to_validate)

        return f(*args)

    return wrapper
