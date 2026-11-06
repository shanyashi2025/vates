from enum import Enum, unique

@unique
class LongShort(Enum):
    """Enum for long or short position."""
    LONG = "LONG"
    SHORT = "SHORT"

@unique
class CallPut(Enum):
    """Enum for call or put option."""
    CALL = "CALL"
    PUT = "PUT"