import os
from dotenv import load_dotenv
from enum import Enum, auto

load_dotenv()

class CheckMode(Enum):
    BYPASS = auto()
    WARN = auto()
    ERROR = auto()

GLOBAL_CHECK_MODE = CheckMode[os.getenv("VATES_GLOBAL_CHECK_MODE", "BYPASS").upper()]
