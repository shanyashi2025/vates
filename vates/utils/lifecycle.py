"""Lightweight per-object lifecycle tracking for sequence checks.

A domain defines its own ``Enum`` of phases (one per object family) and guards its
methods with :func:`transition` and/or the ``require*`` methods below.

Example:
    from enum import Enum, auto

    class AssetPhase(Enum):
        CREATED = auto()
        ROLLED = auto()
        CLOSED = auto()

    class Asset:
        def __init__(self, *, time: int) -> None:
            self.time = time
            # A freshly constructed/spawned asset is CREATED at the current tick.
            self._lc = Lifecycle[AssetPhase](
                created_at=time, created_phase=AssetPhase.CREATED)

        @transition(require_not=AssetPhase.ROLLED, mark=AssetPhase.ROLLED)
        def roll_forward(self) -> None:
            ...  # run-off and rebalance both use this; no double-roll.

        @transition(require_any={AssetPhase.ROLLED, AssetPhase.CREATED},
                    mark=AssetPhase.CLOSED)
        def close_dealing(self) -> None:
            ...  # works for rolled assets *and* assets spawned this period.

Notes:
    :func:`transition` resolves the check level at *decoration time*. When the
    resolved level is ``CheckLevel.BYPASS`` it returns the decorated function
    unchanged, so production pays no dispatch cost and no phases are recorded.
    This assumes ``CHECK_LEVEL`` is fixed at import (it is read from the
    environment once); do not toggle it at runtime, and do not branch production
    logic on ``did``/``snapshot``.
"""

from __future__ import annotations

import functools
import warnings
from enum import Enum
from typing import Generic, Iterable, TypeVar

from vates.global_conf import CHECK_LEVEL, CheckLevel

__all__ = ["Lifecycle", "transition"]

T = TypeVar("T", bound=Enum)


class Lifecycle(Generic[T]):
    """Tracks the last period in which each phase was performed.

    The mapping is keyed by the phase ``Enum`` of the owning domain, so it stays
    bounded by the number of enum members (a handful) regardless of how many
    periods are projected. No per-period reset is needed: a stale timestamp
    (``ROLLED -> t - 1``) naturally fails an ``== t`` test.

    Args:
        created_at (int): Period at which the object came into existence.
        created_phase (T | None): Phase recorded as performed at ``created_at``,
            typically ``Phase.CREATED``. ``None`` records nothing.
    """
    __slots__ = ("_last",)

    def __init__(self, *, created_at: int | None = None, created_phase: T | None = None) -> None:
        self._last: dict[T, int] = {}
        if created_phase is not None:
            self._validate_time(created_at)
            self._validate_phase(created_phase)
            self._last[created_phase] = created_at

    # ------------------------------------------------------------------ queries
    def did(self, phase: T, t: int) -> bool:
        """Return True if ``phase`` was performed at period ``t``."""
        self._validate_phase(phase)
        self._validate_time(t)
        return self._last.get(phase) == t

    def last(self, phase: T) -> int | None:
        """Return the last period ``phase`` was performed at, or None."""
        self._validate_phase(phase)
        return self._last.get(phase)

    @property
    def snapshot(self) -> dict[T, int]:
        """Return a copy of the current ``{phase: last_period}`` mapping."""
        return dict(self._last)

    # --------------------------------------------------------------- transition
    def mark(self, phase: T, t: int) -> None:
        """Record ``phase`` as performed at period ``t``."""
        self._validate_phase(phase)
        self._validate_time(t)
        self._last[phase] = t

    # ------------------------------------------------------------------ guards
    def require_all(self, phases: T | Iterable[T], t: int, *,
                    check_level: CheckLevel | None = None) -> None:
        """Assert that every phase in ``phases`` was performed at period ``t``.
        """
        self._require(self._normalize(phases), t, all_=True, check_level=check_level)

    def require_any(self, phases: T | Iterable[T], t: int, *,
                    check_level: CheckLevel | None = None) -> None:
        """Assert that at least one phase in ``phases`` was performed at ``t``."""
        self._require(self._normalize(phases), t, all_=False, check_level=check_level)

    def require_not_yet(self, phase: T, t: int, *,
                        check_level: CheckLevel | None = None) -> None:
        """Assert that ``phase`` has NOT already been performed at period ``t``.

        Guards against a phase being performed twice within the same period
        (e.g. rolling an asset forward twice).
        """
        self._validate_phase(phase)
        self._validate_time(t)

        level = self._resolve_level(check_level)
        if level is CheckLevel.BYPASS:
            return
        if self._last.get(phase) != t:
            return

        self._emit(f"Phase {phase.name} already performed at t={t}.", level)

    # ------------------------------------------------------------------- dunder
    def __repr__(self) -> str:
        body = ", ".join(f"{phase.name}: {t}" for phase, t in self._last.items())
        return f"Lifecycle({body})"

    # ---------------------------------------------------------------- internals
    def _require(self, phases: tuple[T, ...], t: int, *, all_: bool,
                 check_level: CheckLevel | None) -> None:
        self._validate_time(t)
        level = self._resolve_level(check_level)
        if level is CheckLevel.BYPASS:
            return

        if all_:
            ok = all(self._last.get(phase) == t for phase in phases)
        else:
            ok = any(self._last.get(phase) == t for phase in phases)
        if ok:
            return

        quantifier = "all of" if all_ else "one of"
        self._emit(f"Expected {quantifier} {self._describe(phases)} at t={t}; "
                   f"actual {self._describe_actual()}.", level)

    @classmethod
    def _normalize(cls, phases: T | Iterable[T]) -> tuple[T, ...]:
        if isinstance(phases, Enum):
            items = (phases,)
        else:
            try:
                items = tuple(phases)
            except TypeError:
                raise TypeError(f"Invalid type of 'phases': {type(phases).__name__}, "
                                f"expected 'Enum' or an iterable of 'Enum'.") from None
        if not items:
            raise ValueError("'phases' must not be empty.")
        for phase in items:
            cls._validate_phase(phase)
        return items

    @staticmethod
    def _resolve_level(check_level: CheckLevel | None) -> CheckLevel:
        level = CHECK_LEVEL if check_level is None else check_level
        if not isinstance(level, CheckLevel):
            raise TypeError(f"Invalid type of 'check_level': {type(level).__name__}, "
                            f"expected 'CheckLevel'.")
        return level

    @staticmethod
    def _emit(message: str, level: CheckLevel) -> None:
        if level is CheckLevel.WARN:
            warnings.warn(message, stacklevel=3)
            return
        raise ValueError(message)

    @staticmethod
    def _validate_phase(phase: object) -> None:
        if not isinstance(phase, Enum):
            raise TypeError(f"Invalid type of 'phase': {type(phase).__name__}, "
                            f"expected 'Enum'.")

    @staticmethod
    def _validate_time(t: object) -> None:
        if not isinstance(t, int) or isinstance(t, bool):
            raise TypeError(f"Invalid type of 't': {type(t).__name__}, expected 'int'.")

    @staticmethod
    def _describe(phases: Iterable[T]) -> str:
        return "{" + ", ".join(sorted(phase.name for phase in phases)) + "}"

    def _describe_actual(self) -> str:
        if not self._last:
            return "{}"
        return "{" + ", ".join(f"{phase.name}: {t}" for phase, t in self._last.items()) + "}"


def transition(*, require_all: T | Iterable[T] | None = None,
               require_any: T | Iterable[T] | None = None,
               require_not: T | None = None, require_offset: int = 0,
               mark: T | None = None, lifecycle_attr: str = "_lc",
               check_level: CheckLevel | None = None):
    """Enforce and record a :class:`Lifecycle` transition around a method.

    The decorated method must live on an object exposing the current period as
    ``obj.time`` and a :class:`Lifecycle` at ``obj.<lifecycle_attr>`` (default
    ``obj._lc``).

    When the resolved check level is ``CheckLevel.BYPASS`` this returns the
    function unchanged (no wrapper, no phase recording). The level is resolved
    once, at decoration time.

    Args:
        require_all (T | Iterable[T] | None): Phase(s) that must all have happened
            at ``obj.time + require_offset``.
        require_any (T | Iterable[T] | None): Phase(s) of which at least one must
            have happened at ``obj.time + require_offset``. Mutually exclusive
            with ``require_all``.
        require_not (T | None): Phase that must NOT have happened at ``obj.time``
            (double-call guard).
        require_offset (int): Offset applied to the current period for the
            requirements, defaults to 0. Use ``-1`` for "previous period".
        mark (T | None): Phase recorded at ``obj.time`` after the method returns.
        lifecycle_attr (str): Attribute holding the :class:`Lifecycle`,
            defaults to ``"_lc"``.
        check_level (CheckLevel | None): Override for the check level, defaulting
            to the global ``CHECK_LEVEL`` resolved at decoration time.
    """
    if require_all is not None and require_any is not None:
        raise ValueError("Only one of 'require_all' and 'require_any' may be given.")

    level = Lifecycle._resolve_level(check_level)
    if level is CheckLevel.BYPASS:
        def identity(func):
            return func
        return identity

    def decorator(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            lifecycle = getattr(obj, lifecycle_attr)
            t = obj.time

            if require_all is not None:
                lifecycle.require_all(require_all, t + require_offset, check_level=level)
            if require_any is not None:
                lifecycle.require_any(require_any, t + require_offset, check_level=level)
            if require_not is not None:
                lifecycle.require_not_yet(require_not, t, check_level=level)

            result = func(obj, *args, **kwargs)

            if mark is not None:
                lifecycle.mark(mark, t)
            return result

        return wrapper

    return decorator
