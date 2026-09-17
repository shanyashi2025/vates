from enum import Enum, auto

class StrictnessLevel(Enum):
    BYPASS = auto()
    WARN = auto()
    ERROR = auto()

STRICTNESS_LEVEL = StrictnessLevel.BYPASS
