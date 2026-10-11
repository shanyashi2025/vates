import inspect
from typing import Literal

from vates.global_conf import FALLBACK_START_YEAR
from vates._core.proj_model_engine import ProjModelEngine
from vates._core._time_synchronizer import ProjectionTimeSynchronizer
from vates._core._proj_variables import ConstVariable, TDimVariable


class _FALLBACK:
    engine: ProjModelEngine
    synchronizer: ProjectionTimeSynchronizer

def _make_fallback_engine(start_year: int | None = None) -> ProjModelEngine:
    start_year = start_year if start_year is not None else FALLBACK_START_YEAR
    engine = ProjModelEngine(slug="fallback").configure_run(start_year=start_year, end_year=start_year + 10)
    setattr(_FALLBACK, "engine", engine)
    setattr(_FALLBACK, "synchronizer", engine.time_synchronizer)
    return engine

def get_fallback_engine() -> ProjModelEngine:
    return getattr(_FALLBACK, "engine", None) or _make_fallback_engine()

def get_fallback_synchronizer() -> ProjectionTimeSynchronizer:
    return getattr(_FALLBACK, "synchronizer", None) or _make_fallback_engine().time_synchronizer


def time_synchronized(key: str, /):
    """Attach the instance as a listener of `time_synchronizer`, and add two attributes `time` and `period`.

    - `time_synchronizer` (a ProjectionTimeSynchronizer instance object):
    - `time`, `period`: cached on the instance as plain attributes and refreshed
      on every time change.

    Args:
        key: the `time_synchronizer` string, e.g. "model_engine.time_synchronizer"

    Returns: the decorated class

    """
    if not isinstance(key, str):
        raise TypeError(f"Invalid type of key: '{type(key)}', expected 'str'.")

    if "." not in key:
        engine_param, sync_param = None, key
    else:
        engine_param, sync_param = key.split(".")

    def decorator(cls):
        orig_init = cls.__init__
        if orig_init is object.__init__:
            raise AttributeError(f"{cls.__name__}: '__init__' is not defined.")
        sig = inspect.signature(cls.__init__)
        req_param = engine_param if engine_param is not None else sync_param
        if req_param not in sig.parameters:
            raise KeyError(f"time_synchronizer key '{key}': '{req_param}' does't match signaure of '__init__'.")

        def new_init(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            if engine_param is not None:
                engine = bound.arguments.get(engine_param)
                synchronizer = getattr(engine, sync_param, None)
            else:
                synchronizer = bound.arguments.get(sync_param)
            if synchronizer is None:
                synchronizer = get_fallback_synchronizer()
            if synchronizer is None:
                raise ValueError(f"Failed to get projection time synchronizer.")

            self.time = synchronizer.time
            self.period = synchronizer.period

            orig_init(self, *args, **kwargs)

            synchronizer.attach_listener(self)

        cls.__init__ = new_init
        return cls

    return decorator


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
    if variable_type.lower() not in ("const", "c", "tdim", "t", "time_dimensioned"):
        raise ValueError(f"Invalid value of 'variable_type': '{variable_type}', expected ('const', 'tdim').")

    model_engine = model_engine or get_fallback_engine()

    if variable_type.lower() in ("const", "c"):
        var = ConstVariable(name, owner=owner, group=group, dims=dims)
    else:
        var = TDimVariable(name, owner=owner, group=group, dims=dims,
                           max_t=model_engine.MAX_T, start_date=model_engine.START_DATE)

    model_engine.attach_proj_variable(var)

    return var
