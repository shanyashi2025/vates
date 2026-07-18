import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable


@add_projection_time_synchronizer
class EquityIndex:
    """
    Represents an equity index and its time series.

    Attributes:
        index_id (str): Equity index identifier.
        tdv_tot_return_index (TDimVariable): Total return index.
        tdv_dividend_yield_ac (float): Dividend yield (annual compounding).
    """
    time: int           # for type hint only, will be injected by decorator `has_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `has_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_last_update',
                 'index_id', '_total_return', '_capital_growth', '_dividend_yield', '_dividend_yield_ac', '_total_return_index',
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
        self._total_return: float | None = None
        self._capital_growth: float | None = None
        self._dividend_yield: float | None = None
        self._dividend_yield_ac: float | None = None
        self._total_return_index: float | None = None
        self._last_update: int | None = None

        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=index_id, group='equity_index')
        self.tdv_tot_return_index: TDimVariable = create_tdv("tot_return_index")
        self.tdv_dividend_yield_ac: TDimVariable = create_tdv("dividend_yield_ac")

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    @property
    def total_return(self) -> float | None:
        """float: Total return in period."""
        return self._total_return

    @property
    def capital_growth(self) -> float | None:
        """float: Capital growth in period."""
        return self._capital_growth

    @property
    def dividend_yield(self) -> float | None:
        """float: Dividend yield (monthly) in period."""
        return self._dividend_yield

    @property
    def dividend_yield_ac(self) -> float | None:
        """float: Dividend yield (annual compounding) in period."""
        return self._dividend_yield_ac

    @property
    def total_return_index(self) -> float | None:
        """float: Total return index."""
        return self._total_return_index

    @property
    def arr_tot_return_index(self) -> TDimVariable:
        """float: Total return index."""
        return self.tdv_tot_return_index

    @property
    def arr_dividend_yield_ac(self) -> TDimVariable:
        """TDepVariable: Dividend yield (annual compounding)."""
        return self.tdv_dividend_yield_ac

    def compound_growth_on_update(self) -> None:
        """
        Apply compound growth on the total equity index.
        """
        t = self.time
        self._total_return_index *= 1 + self._total_return
        self.tdv_tot_return_index[t] = self._total_return_index
        self.tdv_dividend_yield_ac[t] = self._dividend_yield_ac
        self._last_update = t

    def update(self, total_return_index: float, dividend_yield_ac: float) -> None:
        """
        Update the equity index values for the current time step.

        Args:
            total_return_index (float): New total return index.
            dividend_yield_ac (float): New dividend yield (annual compounding).
        """
        t = self.time

        self._dividend_yield_ac = dividend_yield_ac
        self._dividend_yield = (1 + dividend_yield_ac) ** (1 / 12) - 1

        if self._total_return_index is None:
            pass
        elif self._total_return_index == 0:
            raise ZeroDivisionError(f"{self.index_id}: can not calculate total return in period as previous index is 0.")
        else:
            self._total_return = total_return_index / self._total_return_index - 1
            self._capital_growth = self._total_return - self._dividend_yield

        self._total_return_index = total_return_index
        self.tdv_tot_return_index[t] = total_return_index
        self.tdv_dividend_yield_ac[t] = dividend_yield_ac

        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.index_id}'"
