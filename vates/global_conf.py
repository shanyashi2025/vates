from enum import Enum, auto

class StrictnessLevel(Enum):
    BYPASS = auto()
    WARN = auto()
    ERROR = auto()

STATE_CHECK_STRICTNESS_LEVEL = StrictnessLevel.ERROR
