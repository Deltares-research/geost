class ValidationSettings:
    """
    Settings for validation behavior.

    VERBOSE: If True, validation errors will be printed to the console.
    DROP_INVALID: If True, invalid rows will be dropped from the DataFrame.
    AUTO_ALIGN: If True, collection headers and data tables will be automatically aligned.
    """

    def __init__(self):
        self.VERBOSE = True
        self.DROP_INVALID = True
        self.AUTO_ALIGN = True


validation = ValidationSettings()
