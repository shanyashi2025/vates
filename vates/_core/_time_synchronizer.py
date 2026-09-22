import pandas as pd
import weakref


class ProjectionTimeSynchronizer:
    """Publishes the current projection `time`/`period` to registered observers.

    `time` and `period` are plain attributes (updated by `set`/`elapse`) so that
    observers are kept in sync by direct assignment, avoiding a property lookup
    on every access. Observers are stored in a `weakref.WeakSet`, which makes
    attachment O(1) and drops dead observers automatically.
    """

    def __init__(self, time: int | None = None, period: pd.Period | None = None):
        self.time: int | None = time
        self.period: pd.Period | None = period
        self._time_observers: weakref.WeakSet = weakref.WeakSet()

    def set(self, *, time: int | None = None, period: pd.Period | None = None) -> None:
        if time is not None:
            self.time = time
        if period is not None:
            self.period = period if isinstance(period, pd.Period) else pd.Period(period, freq="M")
        self._notify_on_time_change()

    def elapse(self, n: int = 1, /) -> None:
        if self.time is not None:
            self.time += n
        if self.period is not None:
            self.period += n
        self._notify_on_time_change()

    def attach_time_observer(self, observer, /) -> None:
        if isinstance(observer, weakref.ref):
            observer = observer()
        # WeakSet handles duplicate detection and dead-reference cleanup.
        if observer is not None:
            self._time_observers.add(observer)

    def detach_time_observer(self, observer, /) -> None:
        if isinstance(observer, weakref.ref):
            observer = observer()
        if observer is not None:
            self._time_observers.discard(observer)

    def _notify_on_time_change(self) -> None:
        time, period = self.time, self.period
        for observer in self._time_observers:
            observer.time = time
            observer.period = period
            hook = getattr(observer, "_update_on_time_change", None)
            if hook is not None:
                hook()


FALLBACK_TIME_SYNCHRONIZER: ProjectionTimeSynchronizer | None = None

def add_projection_time_synchronizer(_cls=None, /):
    """Attach the instance as an observer of `time_synchronizer`, and add two attributes `time` and `period`.

    - `time_synchronizer` (a ProjectionTimeSynchronizer instance object):
                    ┌────────────────────────────────────────────────────────┐
                    │ if `model_engine` (usually a ProjModelEngine instance  │
                    │ object) exists in **kwargs, and is not None            │
                    └────────────────────────────┬───────────────────────────┘
                            ┌─────── True ───────├──────── False ─────────┐
                            │                                             │
            ┌───────────────▼────────────────┐                            │
            │ if `model_engine` hasattr      ├─────── False ────────┐     │
            │    `time_synchronizer`         │                      │     │
            └───────────────┬────────────────┘                      │     │
                          True                                      │     │
                            │                                       │     │
            ┌───────────────▼───────────────────┐                   │     │
            │ if model_engine.time_synchronizer ├──── False ────┐   │     │
            │          is not None              │               │   │     │
            └───────────────┬───────────────────┘               │   │     │
                          True                                  │   │     │
                            │                                   │   │     │
            ┌───────────────▼────────────────┐         ┌────────▼───▼─────▼─────────┐
            │ model_engine.time_synchronizer │         │ FALLBACK_TIME_SYNCHRONIZER │
            └────────────────────────────────┘         └────────────────────────────┘

    - `time`, `period`: cached on the instance as plain attributes and refreshed
      on every time change.

    Args:
        _cls: the class object to be decorated

    Returns: the decorated class

    """

    def decorator(cls):
        original_init = getattr(cls, "__init__", None)

        def new_init(self, *args, **kwargs):
            model_engine = kwargs.get("model_engine")
            time_synchronizer = None
            if model_engine is not None:
                if hasattr(model_engine, "time_synchronizer"):
                    time_synchronizer = getattr(model_engine, "time_synchronizer")
            if time_synchronizer is None:
                if FALLBACK_TIME_SYNCHRONIZER is not None:
                    time_synchronizer = FALLBACK_TIME_SYNCHRONIZER
                else:
                    raise ValueError(f"Failed to add projection time synchronizer.")

            self.time = time_synchronizer.time
            self.period = time_synchronizer.period

            if original_init and original_init is not object.__init__:
                original_init(self, *args, **kwargs)

            time_synchronizer.attach_time_observer(self)

        cls.__init__ = new_init
        return cls

    if _cls is None:
        return decorator

    return decorator(_cls)
