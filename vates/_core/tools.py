import inspect
from typing import Literal, overload

from vates._core.proj_model_engine import ProjModelEngine
from vates._core._time_synchronizer import ProjectionTimeSynchronizer
from vates._core._proj_variables import make_variable, ScalarVariable, ConstantArrayVariable, TDimVariable, TDimArrayVariable


class _FALLBACK:
    start_year: int = 2025
    end_year: int = 2045
    engine: ProjModelEngine
    synchronizer: ProjectionTimeSynchronizer

    @classmethod
    def make_engine(cls) -> ProjModelEngine:
        engine = ProjModelEngine(slug="fallback").configure_run(start_year=cls.start_year, end_year=cls.end_year)
        engine.time = 0  # let the engine and synchronizer have `time` and `period`
        cls.engine = engine
        cls.synchronizer = engine._time_synchronizer
        return engine

def get_fallback_engine() -> ProjModelEngine:
    return getattr(_FALLBACK, "engine", None) or _FALLBACK.make_engine()

def get_fallback_synchronizer() -> ProjectionTimeSynchronizer:
    return getattr(_FALLBACK, "synchronizer", None) or _FALLBACK.make_engine()._time_synchronizer

def time_synchronized(key: str, /):
    """Keep instances of the decorated class synchronized with projection time.

    The key names an ``__init__`` parameter followed by zero or more attributes,
    separated by dots. The resolved object must expose ``time``, ``period``, and
    ``attach_time_listener``; this can be a ``ProjModelEngine`` or a
    ``ProjectionTimeSynchronizer``.

    Args:
        key: A path such as ``"model_engine"`` or
            ``"context.model.engine"``. If the root argument is ``None``, the
            fallback synchronizer is used. If a non-``None`` root is supplied
            but a later attribute cannot be resolved, an ``AttributeError`` is
            raised.

    Returns:
        The decorated class. Its instances receive cached ``time`` and
        ``period`` attributes and are registered after successful
        initialization.

    """
    if not isinstance(key, str):
        raise TypeError(f"Invalid type of key: '{type(key)}', expected 'str'.")

    params = key.split(".")

    def decorator(cls):
        orig_init = cls.__init__
        if orig_init is object.__init__:
            raise AttributeError(f"{cls.__name__}: '__init__' is not defined.")
        sig = inspect.signature(cls.__init__)
        if params[0] not in sig.parameters:
            raise KeyError(f"time_synchronizer key '{key}': '{params[0]}' doesn't match signature of '__init__'.")

        def new_init(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            synchronizer = bound.arguments.get(params[0])
            if synchronizer is None:
                synchronizer = get_fallback_synchronizer()
            else:
                for param in params[1:]:
                    try:
                        synchronizer = getattr(synchronizer, param)
                    except AttributeError as exc:
                        raise AttributeError(
                            f"time_synchronized key '{key}': cannot resolve "
                            f"'{param}' on '{type(synchronizer).__name__}'."
                        ) from exc
                    if synchronizer is None:
                        raise AttributeError(
                            f"time_synchronized key '{key}': '{param}' resolved to None."
                        )
            if synchronizer is None:
                raise ValueError(f"Failed to get projection time synchronizer.")

            self.time = synchronizer.time
            self.period = synchronizer.period

            orig_init(self, *args, **kwargs)

            synchronizer.attach_time_listener(self)

        cls.__init__ = new_init
        return cls

    return decorator


@overload
def make_proj_variable(
    name: str,
    /,
    *,
    model_engine: ProjModelEngine | None = None,
    variable_type: Literal["constant", "c"],
    owner: str,
    group: str,
) -> ScalarVariable:
    ...

@overload
def make_proj_variable(
    name: str,
    /,
    *,
    model_engine: ProjModelEngine | None = None,
    variable_type: Literal["constant", "c"],
    owner: str,
    group: str,
    dims: list | tuple,
) -> ConstantArrayVariable:
    ...

@overload
def make_proj_variable(
    name: str,
    /,
    *,
    model_engine: ProjModelEngine | None = None,
    variable_type: Literal["t", "time_dimensioned"] = "t",
    owner: str,
    group: str,
) -> TDimVariable:
    ...

@overload
def make_proj_variable(
    name: str,
    /,
    *,
    model_engine: ProjModelEngine | None = None,
    variable_type: Literal["t", "time_dimensioned"] = "t",
    owner: str,
    group: str,
    dims: list | tuple,
) -> TDimArrayVariable:
    ...

def make_proj_variable(
    name: str,
    /,
    *,
    model_engine: ProjModelEngine | None = None,
    variable_type: Literal["constant", "c", "tdim", "t", "time_dimensioned"] = "t",
    owner: str,
    group: str,
    dims: list | tuple | None = None,
) -> ScalarVariable | ConstantArrayVariable | TDimVariable | TDimArrayVariable:
    if not isinstance(variable_type, str):
        raise TypeError(f"Invalid type of 'variable_type': '{type(variable_type)}', expected 'str'.")
    if variable_type.lower() not in ("constant", "c", "tdim", "t", "time_dimensioned"):
        raise ValueError(f"Invalid value of 'variable_type': '{variable_type}', expected ('constant', 'tdim').")

    model_engine = model_engine or get_fallback_engine()
    max_t = model_engine.MAX_T if not variable_type.lower() in ("constant", "c") else None
    var = make_variable(name, owner=owner, group=group, max_t=max_t, dims=dims)

    return var @ model_engine
