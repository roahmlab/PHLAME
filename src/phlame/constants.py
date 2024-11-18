from enum import Enum


MINIMUM_S_SPAN = 1e-6

class ResultType(Enum):
    SUCCESS = 1
    TIMEOUT = 2
    CVODE_ERROR = 3
    OTHER_ERROR = 4
