import warnings
from functools import wraps


class ValidationWarning(UserWarning):
    """Warning for when validation fails."""

    pass


class AlignmentWarning(UserWarning):
    """Warning for when header and data are not aligned."""

    pass


def future_deprecation_warning(alternative_func=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"The function {func.__name__} is deprecated and will be removed in a future version."
            if alternative_func:
                message += f" Please use {alternative_func.__name__} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
