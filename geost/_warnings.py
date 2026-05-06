import warnings
from functools import wraps


class ValidationWarning(UserWarning):
    """Warning for when validation fails."""

    pass


class AlignmentWarning(UserWarning):
    """Warning for when header and data are not aligned."""

    pass


class MixedDepthWarning(UserWarning):
    """Warning for when depth columns contain a mix of positive and negative values."""

    pass


def future_deprecation_warning(alternative_func=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"The GeoST function {func.__name__} is deprecated and will be removed in a future version."
            if alternative_func:
                message += f" Please use {alternative_func.__name__} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
