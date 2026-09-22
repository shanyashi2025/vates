import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable
from vates.utils import maybe_raise_if_ne


@add_projection_time_synchronizer
class Currency:
    """
    Represents a currency and its FX rate time series.

    Attributes:
        currency_id (str): Currency identifier.
        tdv_fx_rate (TDimVariable): Current FX rate.
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    _fx_rate: float
    _fx_rate_prev: float

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', "_last_update",
                 'currency_id', '_fx_rate', '_fx_rate_prev', 'tdv_fx_rate', )

    def __init__(
        self,
        currency_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
    ) -> None:
        """
        Initialize a Currency object.

        Args:
            currency_id (str): Currency identifier.
        """
        self.currency_id: str = currency_id
        self._fx_rate = 1.0
        self.tdv_fx_rate: TDimVariable = TDimVariable("fx_rate", model_engine=model_engine, owner=currency_id, group='currency')
        self._last_update: int = self.time or 0

    @property
    def fx_rate(self) -> float:
        """float: Current FX rate"""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._fx_rate

    @property
    def fx_rate_prev(self) -> float:
        """float: Previous FX rate"""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._fx_rate_prev

    @property
    def arr_fx_rate(self) -> TDimVariable:
        """TDimVariable: FX rates."""
        return self.tdv_fx_rate

    def update(self, *, fx_rate: float = None, is_apply_compound_growth: bool = False) -> None:
        """
        Update the FX rate for the current time step.

        Args:
            fx_rate (float): New FX rate.
            is_apply_compound_growth (bool): True if applying compound growth, defaults to False.
        """
        if is_apply_compound_growth:
            if fx_rate is not None:
                raise ValueError(f"is_apply_compound_growth=True but other fx_rate is given.")
            if self._fx_rate_prev == 0:
                raise ZeroDivisionError(f"{self.currency_id}: previous fx rate is zero.")
            growth = self._fx_rate / self._fx_rate_prev
            self._fx_rate_prev = self._fx_rate
            self._fx_rate *= growth
        else:
            if fx_rate is None:
                raise ValueError(f"fx_rate is None.")
            self._fx_rate, self._fx_rate_prev = fx_rate, self._fx_rate

        t = self.time
        self.tdv_fx_rate[t] = fx_rate
        self._last_update = ("updated", t)

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.currency_id}'"
