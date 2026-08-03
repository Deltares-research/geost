class GeostError(Exception):
    """Base class for all geost exceptions."""


class ModelError(GeostError):
    """Base class for all geost model exceptions."""


class InvalidModelError(ModelError):
    """Raised when a model is invalid."""
