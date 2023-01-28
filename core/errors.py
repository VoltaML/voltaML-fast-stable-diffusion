class DimensionError(Exception):
    "Raised when the dimension of the input is not correct"


class ModelFailedError(Exception):
    "Raised when the model fails to generate and image or fails to load"


class ModelNotLoadedError(Exception):
    "Raised when the model is blocked from being loaded automatically"


class BadSchedulerError(Exception):
    "Raised when the scheduler is not of a proper type"


class InferenceInterruptedError(Exception):
    "Raised when the model is interrupted"
