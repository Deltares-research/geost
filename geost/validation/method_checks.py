from functools import wraps


def _requires_geometry(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.has_geometry:
            raise TypeError(
                f"Method '{func.__name__}' requires a GeoDataFrame with a valid geometry column."
            )
        return func(self, *args, **kwargs)

    return wrapper


def _requires_depth(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.has_depth_columns:
            raise KeyError(  # TODO: Check formatting of this error message
                f"Method '{func.__name__}' requires depth information in the DataFrame. "
                "Please ensure that the DataFrame contains one of the following the "
                "required combinations of depth columns:\n"
                " - 'surface', 'top' and 'bottom'\n"
                " - 'surface' and 'bottom'\n"
                " - 'surface' and 'depth'\n"
            )
        return func(self, *args, **kwargs)

    return wrapper
