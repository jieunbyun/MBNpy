import warnings
from .inference import *

warnings.warn(
    "operation.py is deprecated and replaced with inference.py. Please update your imports.",
    DeprecationWarning,
    stacklevel=2
)
