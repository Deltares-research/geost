import warnings
from functools import wraps
from typing import Callable


class ValidationWarning(UserWarning):
    """Warning for when validation fails."""

    pass


class AlignmentWarning(UserWarning):
    """Warning for when header and data are not aligned."""

    pass


class MixedDepthWarning(UserWarning):
    """Warning for when depth columns contain a mix of positive and negative values."""

    pass


def future_deprecation_warning(
    alternative_func: Callable | list[Callable] | None = None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"The GeoST function {func.__name__} is deprecated and will be removed in a future version."
            if alternative_func:
                if isinstance(alternative_func, list):
                    alt_names = ", ".join(func.__name__ for func in alternative_func)
                    message += f" Please use one of {alt_names} instead."
                else:
                    message += f" Please use {alternative_func.__name__} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
