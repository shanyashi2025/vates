import os
from dotenv import load_dotenv
from enum import Enum, auto

load_dotenv()

class CheckLevel(Enum):
    BYPASS = auto()
    WARN = auto()
    ERROR = auto()

CHECK_LEVEL = CheckLevel[os.getenv("VATES_CHECK_LEVEL", "BYPASS").upper()]
