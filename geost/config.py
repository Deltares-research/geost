class ValidationSettings:
    """
    Settings for validation behavior.

    VERBOSE: If True, validation errors will be printed to the console.
    DROP_INVALID: If True, invalid rows will automatically be dropped from (Geo)DataFrames.
    FLAG_INVALID: If True, invalid rows will be flagged in (Geo)DataFrames. Note: only works if DROP_INVALID is False.
    AUTO_ALIGN: If True, collection headers and data tables will automatically be aligned.
    """

    VERBOSE = True
    DROP_INVALID = True
    FLAG_INVALID = False
    AUTO_ALIGN = True


validation = ValidationSettings()
