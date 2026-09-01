class GeostError(Exception):
    """Base class for all geost exceptions."""


class MissingCRSError(GeostError):
    """Raised when a coordinate reference system (CRS) is missing."""


class ModelError(GeostError):
    """Base class for all geost model exceptions."""


class InvalidModelError(ModelError):
    """Raised when a model is invalid."""


class ModelTypeError(ModelError):
    """Raised when a model is of the wrong type."""


class MissingUnitError(ModelError):
    """Raised when a unit is missing in the model."""
