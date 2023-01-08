class DimensionError(Exception):
    "Raised when the dimension of the input is not correct"


class ModelFailedError(Exception):
    "Raised when the model fails to generate and image or fails to load"


class AutoLoadDisabledError(Exception):
    "Raised when the model is blocked from being loaded automatically"
