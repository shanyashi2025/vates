from enum import Enum, unique

@unique
class LongOrShort(Enum):
    """Enum for long or short position."""
    LONG = "LONG"
    SHORT = "SHORT"

@unique
class CallOrPut(Enum):
    """Enum for call or put option."""
    CALL = "CALL"
    PUT = "PUT"