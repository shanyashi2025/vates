from importlib.metadata import version
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

__version__ = version("vates")

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