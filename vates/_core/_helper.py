import inspect
import json
import os
import pandas as pd
import sys
from typing import Literal
from pathlib import Path

def get_environ() -> dict[Literal["COMPUTERNAME", "USERNAME", "USERDOMAIN", "INTERPRETER"], str | None]:
    return {
        "COMPUTERNAME": os.getenv("COMPUTERNAME"),
        "USERNAME": os.getenv("USERNAME"),
        "USERDOMAIN": os.getenv("USERDOMAIN"),
        "INTERPRETER": sys.executable,
    }

def get_f_filename_lineno() -> tuple[str, int]:
    frame = inspect.currentframe().f_back
    filename = os.path.abspath(frame.f_code.co_filename)
    lineno = frame.f_lineno
    return filename, lineno

def search_file_in_dirs(filename: str, dirs: list[Path] | tuple[Path, ...]) -> list[Path | None]:
    """Return `list[Path]`: All matches per `dirs` order; `[None]` if no matches."""
    filepath = Path(filename)
    if filepath.is_absolute() and filepath.is_file():  # absolute path
        return [filepath]
    matches = []
    for directory in dirs:  # search in order
        filepath = (directory / filename).resolve()
        if filepath.is_file():
            matches.append(filepath)
    return matches if matches else [None]
