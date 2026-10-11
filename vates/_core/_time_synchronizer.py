import pandas as pd
import weakref


class ProjectionTimeSynchronizer:
    """Dispatches the current projection `time`/`period` to registered listeners.

    `time` and `period` are plain attributes (updated by `set`/`elapse`) so that
    listeners are kept in sync by direct assignment, avoiding a property lookup
    on every access. Listeners are stored in a `weakref.WeakSet`, which makes
    attachment O(1) and drops dead listeners automatically.
    """

    def __init__(self, time: int | None = None, period: pd.Period | None = None):
        self.time: int | None = time
        self.period: pd.Period | None = period
        self._listeners: weakref.WeakSet = weakref.WeakSet()

    def set(self, *, time: int | None = None, period: pd.Period | None = None) -> None:
        if time is not None:
            self.time = time
        if period is not None:
            self.period = period if isinstance(period, pd.Period) else pd.Period(period, freq="M")
        self._emit_time_change()

    def elapse(self, n: int = 1, /) -> None:
        if self.time is not None:
            self.time += n
        if self.period is not None:
            self.period += n
        self._emit_time_change()

    def attach_listener(self, listener, /) -> None:
        if isinstance(listener, weakref.ref):
            listener = listener()
        # WeakSet handles duplicate detection and dead-reference cleanup.
        if listener is not None:
            self._listeners.add(listener)

    def detach_listener(self, listener, /) -> None:
        if isinstance(listener, weakref.ref):
            listener = listener()
        if listener is not None:
            self._listeners.discard(listener)

    def _emit_time_change(self) -> None:
        time, period = self.time, self.period
        for listener in self._listeners:
            listener.time = time
            listener.period = period
            hook = getattr(listener, "_on_time_change", None)
            if hook is not None:
                hook()


