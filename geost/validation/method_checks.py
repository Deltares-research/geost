from functools import wraps


def _requires_geometry(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from geost.accessor import GeostFrame  # Avoid circular imports
        from geost.base import Collection

        if isinstance(self, Collection):
            if not self.header_has_geometry:
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


def _requires_surface(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from geost.accessor import GeostFrame  # Avoid circular imports
        from geost.base import Collection

        if isinstance(self, Collection):
            has_surface = self.data.gst.has_surface_column
        elif isinstance(self, GeostFrame):
            has_surface = self.has_surface_column

        if not has_surface:
            raise KeyError(
                f"Method '{func.__name__}' requires a surface column in the DataFrame. "
                "Please ensure that the DataFrame contains a 'surface' column."
            )
        return func(self, *args, **kwargs)

    return wrapper


def _requires_depth(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from geost.accessor import GeostFrame  # Avoid circular imports
        from geost.base import Collection

        if isinstance(self, Collection):
            has_depth = self.data.gst.has_depth_columns
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


def _requires_xy(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from geost.accessor import GeostFrame  # Avoid circular imports
        from geost.base import Collection

        if isinstance(self, Collection):
            has_xy = self.data.gst.has_xy_columns
        elif isinstance(self, GeostFrame):
            has_xy = self.has_xy_columns

        if not has_xy:
            raise KeyError(  # TODO: Check formatting of this error message
                f"Method '{func.__name__}' requires x, y information in the DataFrame. "
                "Please ensure that the DataFrame contains 'x' and 'y' columns."
            )
        return func(self, *args, **kwargs)

    return wrapper
