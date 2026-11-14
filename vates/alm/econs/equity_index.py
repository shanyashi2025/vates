import pandas as pd

from vates._core import ProjModelEngine, time_synchronized, make_proj_variable
from vates.utils import maybe_raise_on_compare


@time_synchronized("model_engine.time_synchronizer")
class EquityIndex:
    """
    Represents an equity index and its time series.

    Attributes:
        index_id (str): Equity index identifier.

    """
    time: int           # for type hint only, will be injected by decorator `time_synchronized`
    period: pd.Period   # for type hint only, will be injected by decorator `time_synchronized`

    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_last_update', 'index_id',
                 '_total_return_index', '_total_return_index_prev', '_price_return_index', '_price_return_index_prev',
                 'tdv_tot_return_index', 'tdv_tot_price_index', 'tdv_dividend_yield', )

    def __init__(
        self,
        index_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        total_return_index: float = float("nan"),
        price_return_index: float = float("nan"),
    ) -> None:
        """
        Initialize an EquityIndex object.

        Args:
            index_id (str): Equity index identifier.
            total_return_index (float): Total return index.
            price_return_index (float): Total return index.
        """
        self.index_id: str = index_id
        self._total_return_index: float = total_return_index
        self._price_return_index: float = price_return_index
        self._total_return_index_prev: float = float("nan")
        self._price_return_index_prev: float = float("nan")
        self._total_return: float= 0
        self._price_return: float= 0
        tdv_kwargs = {"model_engine": model_engine, "owner": index_id, "group": 'equity_index'}
        self.tdv_tot_return_index = make_proj_variable("total_return_index", **tdv_kwargs)
        self.tdv_pri_return_index = make_proj_variable("price_return_index", **tdv_kwargs)
        self.tdv_dividend_yield = make_proj_variable("dividend_yield", **tdv_kwargs)
        self._last_update: int = self.time or 0

    @property
    def total_return(self) -> float:
        """float: Total return in period."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._total_return

    @property
    def price_return(self) -> float:
        """float: Price return (capital gain or loss) in period."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._price_return

    @property
    def dividend_yield(self) -> float:
        """float: Effective dividend yield over the period."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._total_return - self._price_return

    @property
    def total_return_index(self) -> float:
        """float: Current total return index."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._total_return_index

    @property
    def total_return_index_prev(self) -> float:
        """float: Previous total return index."""
        return self._total_return_index_prev

    @property
    def price_return_index(self) -> float:
        """float: Current price return index."""
        maybe_raise_on_compare(self._last_update, self.time)
        return self._price_return_index

    @property
    def price_return_index_prev(self) -> float:
        """float: Previous price return index."""
        return self._price_return_index_prev

    def _on_time_change(self) -> None:
        self._total_return_index_prev = self._total_return_index
        self._price_return_index_prev = self._price_return_index

    def update(self, *, total_return_index: float = float("nan"), price_return_index: float = float("nan"),
               apply_compound_growth: bool = False) -> None:
        """
        Update the equity index values for the current time step. The decomposition of total return (TR),
        price return (PR, capital gain or loss) and dividend yield (DY) are: TR = PR + DY

        Args:
            total_return_index (float): Total return index (TRI).
            price_return_index (float): Price return index (PRI).
            apply_compound_growth (bool): True if applying compound growth to TRI and PRI, defaults to False.
        """
        if apply_compound_growth:
            self._total_return_index = self._total_return_index_prev * (1 + self._total_return)
            self._price_return_index = self._price_return_index_prev * (1 + self._price_return)
        else:
            self._total_return_index = total_return_index
            self._price_return_index = price_return_index
            self._total_return = self._total_return_index / self._total_return_index_prev - 1
            self._price_return = self._price_return_index / self._price_return_index_prev - 1

        t = self.time
        self.tdv_tot_return_index[t] = self._total_return_index
        self.tdv_pri_return_index[t] = self._price_return_index
        self.tdv_dividend_yield[t] = self._total_return - self._price_return
        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.index_id}'"

    def retrieve_total_return_index(self, t: int) -> float:
        maybe_raise_on_compare(t, must_ge=0, must_le=self.time)
        return self.tdv_tot_return_index[t]

    def retrieve_price_return_index(self, t: int) -> float:
        maybe_raise_on_compare(t, must_ge=0, must_le=self.time)
        return self.tdv_pri_return_index[t]
