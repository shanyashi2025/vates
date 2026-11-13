import pandas as pd

from vates._core import ProjModelEngine, time_synchronized, TDimVariable, make_proj_variable
from vates.utils import maybe_raise_if_ne


@time_synchronized("model_engine.time_synchronizer")
class Currency:
    """
    Represents a currency and its FX rate time series.

    Attributes:
        currency_id (str): Currency identifier.
        tdv_fx_rate (TDimVariable): Current FX rate.
    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', "_last_update",
                 'currency_id', '_fx_rate', '_fx_rate_prev', 'tdv_fx_rate', )

    def __init__(
        self,
        currency_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        fx_rate: float = float("nan"),
    ) -> None:
        """
        Initialize a Currency object.

        Args:
            currency_id (str): Currency identifier.
            fx_rate (float): FX rate.
        """
        self.currency_id: str = currency_id
        self._fx_rate: float = fx_rate
        self._fx_rate_prev: float = float("nan")
        self.tdv_fx_rate = make_proj_variable("fx_rate", model_engine=model_engine, owner=currency_id, group='currency')
        self._last_update: int = self.time or 0

    @property
    def fx_rate(self) -> float:
        """float: Current FX rate"""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._fx_rate

    @property
    def fx_rate_prev(self) -> float:
        """float: Previous FX rate"""
        return self._fx_rate_prev

    @property
    def arr_fx_rate(self) -> TDimVariable:
        """TDimVariable: FX rates."""
        return self.tdv_fx_rate

    def update(self, *, fx_rate: float = float("nan"), apply_compound_growth: bool = False) -> None:
        """
        Update the FX rate for the current time step.

        Args:
            fx_rate (float): New FX rate.
            apply_compound_growth (bool): True if applying compound growth, defaults to False.
        """
        if apply_compound_growth:
            growth = self._fx_rate / self._fx_rate_prev
            self._fx_rate_prev = self._fx_rate
            self._fx_rate *= growth
        else:
            self._fx_rate, self._fx_rate_prev = fx_rate, self._fx_rate

        t = self.time
        self.tdv_fx_rate[t] = fx_rate
        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.currency_id}'"
