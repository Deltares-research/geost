from functools import wraps


def _requires_geometry(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from geost.accessor import GeostFrame  # Avoid circular imports
        from geost.base import Collection

        if isinstance(self, Collection):
            if not self._header_has_geometry:
                raise TypeError(
                    f"Method '{func.__name__}' requires a header with a valid geometry column. "
                    "Use `set_header_from_data` to set the header from the data and ensure "
                    "that the geometry column is properly set."
                )
        elif isinstance(self, GeostFrame):
            if not self.has_geometry:
                raise TypeError(
                    f"Method '{func.__name__}' requires a GeoDataFrame with a valid geometry column."
                )

        return func(self, *args, **kwargs)

    return wrapper


def _requires_depth(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from geost.accessor import GeostFrame  # Avoid circular imports
        from geost.base import Collection

        if isinstance(self, Collection):
            has_depth = self.data.has_depth_columns
        elif isinstance(self, GeostFrame):
            has_depth = self.has_depth_columns

        if not has_depth:
            raise KeyError(  # TODO: Check formatting of this error message
                f"Method '{func.__name__}' requires depth information in the DataFrame. "
                "Please ensure that the DataFrame contains one of the following the "
                "required combinations of depth columns:"
                " - 'surface', 'top' and 'bottom'"
                " - 'surface' and 'bottom'"
                " - 'surface' and 'depth'"
            )
        return func(self, *args, **kwargs)

    return wrapper
