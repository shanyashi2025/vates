from importlib.metadata import version, PackageNotFoundError
from vates._core import (
    ProjModelEngine,
    StochExecutor,
    ConstVariable,
    TDimVariable,
    KeyedArray,
    proj_result,
)
from vates import alm
from vates import finmath
from vates import solvency
from vates import utils

try:
    __version__ = version("vates")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # core
    'ProjModelEngine',
    'StochExecutor',
    'ConstVariable',
    'TDimVariable',
    'KeyedArray',
    'proj_result',
    # utils
    'utils',
    # libs
    'alm',
    'finmath',
    'solvency',

]