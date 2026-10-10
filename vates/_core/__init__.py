import warnings
from typing import Literal

from vates._core.proj_model_engine import ProjModelEngine
from vates._core.stoch_executor import StochExecutor
from vates._core.proj_variables import ConstVariable, TDimVariable, ProjVariable
from vates._core._time_synchronizer import time_synchronized
from vates._core._utils import proj_result


def make_proj_variable(
    name: str,
    /,
    *,
    model_engine: ProjModelEngine | None = None,
    variable_type: Literal["const", "c", "tdim", "t", "time_dimensioned"] = "t",
    owner: str,
    group: str,
    dims: list | None = None,
) -> ConstVariable | TDimVariable:
    if not isinstance(variable_type, str):
        raise TypeError(f"Invalid type of 'variable_type': '{type(variable_type)}', expected 'str'.")
    if model_engine is not None and not isinstance(model_engine, ProjModelEngine):
        raise TypeError(f"Invalid type of 'model_engine': '{type(model_engine)}', expected 'ProjModelEngine'.")

    if variable_type.lower() in ("const", "c"):
        var = ConstVariable(name, owner=owner, group=group, dims=dims)
    elif variable_type.lower() in ("tdim", "t", "time_dimensioned"):
        if model_engine is not None:
            var = TDimVariable(name, owner=owner, group=group, dims=dims, max_t=model_engine.MAX_T,
                               start_date=model_engine.START_DATE)
        else:
            var = TDimVariable(name, owner=owner, group=group, dims=dims, max_t=1200)
            warnings.warn(f"Create 'TDimVariable': 'max_t=1200' and 'start_date=None'; 'model_engine' is 'None'.")
    else:
        raise ValueError(f"Invalid value of 'variable_type': '{variable_type}', expected ('const', 'tdim').")

    if model_engine is not None:
        model_engine.attach_proj_variable(var)

    return var


__all__ = [
    'ProjModelEngine',
    'StochExecutor',
    'make_proj_variable',
    'ConstVariable',
    'TDimVariable',
    'proj_result',
    'time_synchronized',

]