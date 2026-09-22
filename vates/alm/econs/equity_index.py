import pandas as pd

from vates._core import ProjModelEngine, time_synchronized, TDimVariable
from vates.utils import maybe_raise_if_ne


@time_synchronized
class EquityIndex:
    """
    Represents an equity index and its time series.

    Attributes:
        index_id (str): Equity index identifier.
        tdv_tot_return_index (TDimVariable): Total return index.
        tdv_dividend_yield_ac (float): Dividend yield (annual compounding).
    """
    time: int           # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `add_projection_time_synchronizer`
    _dividend_yield_ac: float
    _total_return_index: float
    _total_return_index_prev: float

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update',
                 'index_id', '_dividend_yield_ac', '_total_return_index', '_total_return_index_prev',
                 'tdv_tot_return_index', 'tdv_dividend_yield_ac', )

    def __init__(
        self,
        index_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
    ) -> None:
        """
        Initialize an EquityIndex object.

        Args:
            index_id (str): Equity index identifier.
        """
        self.index_id: str = index_id
        self._total_return_index = 1.0
        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=index_id, group='equity_index')
        self.tdv_tot_return_index: TDimVariable = create_tdv("tot_return_index")
        self.tdv_dividend_yield_ac: TDimVariable = create_tdv("dividend_yield_ac")
        self._last_update: int = self.time or 0

    @property
    def total_return(self) -> float:
        """float: Total return in period."""
        maybe_raise_if_ne(self._last_update, self.time)
        if self._total_return_index_prev == 0:
            raise ZeroDivisionError(f"{self.index_id}: previous total return index is zero.")
        return self._total_return_index / self._total_return_index_prev - 1

    @property
    def capital_growth(self) -> float:
        """float: Capital growth in period."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self.total_return - self.dividend_yield

    @property
    def dividend_yield(self) -> float:
        """float: Dividend yield (monthly) in period."""
        maybe_raise_if_ne(self._last_update, self.time)
        return (1 + self.dividend_yield_ac) ** (1 / 12) - 1

    @property
    def dividend_yield_ac(self) -> float:
        """float: Dividend yield (annual compounding) in period."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._dividend_yield_ac

    @property
    def total_return_index(self) -> float:
        """float: Current total return index."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._total_return_index

    @property
    def total_return_index_prev(self) -> float:
        """float: Previous total return index."""
        maybe_raise_if_ne(self._last_update, self.time)
        return self._total_return_index_prev

    @property
    def arr_tot_return_index(self) -> TDimVariable:
        """TDimVariable: Total return index."""
        return self.tdv_tot_return_index

    @property
    def arr_dividend_yield_ac(self) -> TDimVariable:
        """TDepVariable: Dividend yield (annual compounding)."""
        return self.tdv_dividend_yield_ac

    def update(self, *, total_return_index: float = None, dividend_yield_ac: float = None,
               is_apply_compound_growth: bool = False) -> None:
        """
        Update the equity index values for the current time step.

        Args:
            total_return_index (float): New total return index.
            dividend_yield_ac (float): New dividend yield (annual compounding).
            is_apply_compound_growth (bool): True if applying compound growth on total return index, defaults to False.
        """
        if is_apply_compound_growth:
            if any(x is not None for x in (total_return_index, dividend_yield_ac)):
                raise ValueError(f"is_apply_compound_growth=True but other arguments are given.")
            growth = self._total_return_index / self._total_return_index_prev
            self._total_return_index_prev = self._total_return_index
            self._total_return_index *= growth
        else:
            if any(x is None for x in (total_return_index, dividend_yield_ac)):
                raise ValueError(f"total_return_index or dividend_yield_ac is None.")
            self._dividend_yield_ac = dividend_yield_ac
            self._total_return_index_prev = self._total_return_index
            self._total_return_index = total_return_index

        t = self.time
        self.tdv_tot_return_index[t] = self._total_return_index
        self.tdv_dividend_yield_ac[t] = self._dividend_yield_ac
        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.index_id}'"
