class GeostError(Exception):
    """Base class for all geost exceptions."""


class MissingCRSError(GeostError):
    """Raised when a coordinate reference system (CRS) is missing."""


class PositionalColumnError(GeostError):
    """Raised when a required positional column information is missing."""


class MissingSurveyIDError(PositionalColumnError):
    """Raised when data is missing a survey ID column."""


class MissingDepthError(PositionalColumnError):
    """Raised when data is missing depth information."""


class MissingGeometryError(PositionalColumnError):
    """Raised when data is missing geometry information."""


class MissingSurfaceError(PositionalColumnError):
    """Raised when data is missing surface information."""


class MissingXYError(PositionalColumnError):
    """Raised when data is missing X and Y coordinate information."""


class ModelError(GeostError):
    """Base class for all geost model exceptions."""


class InvalidModelError(ModelError):
    """Raised when a model is invalid."""


class ModelTypeError(ModelError):
    """Raised when a model is of the wrong type."""


class MissingUnitError(ModelError):
    """Raised when a unit is missing in the model."""
