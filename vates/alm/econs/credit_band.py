import numpy as np
import numpy.typing as npt
import pandas as pd

from vates._core import ProjModelEngine, add_projection_time_synchronizer, TDimVariable


@add_projection_time_synchronizer
class CreditBand:
    """
    Represents credit information for a financial instrument.

    Attributes:
        band_id (str): credit band identifier.
        _spread (npt.NDArray[np.float64] | None): Credit spread(s).
        _spotmult (npt.NDArray[np.float64] | None): Spot rate multipliers.
        tdv_prob_of_default_ac (float): Probability of default (annual compounding).
        tdv_recovery_rate (float): Recovery rate.
    """
    time: int           # for type hint only, will be injected by decorator `has_time_synchronizer`
    period: pd.Period   # for type hint only, will be injected by decorator `has_time_synchronizer`
    
    __slots__ = ('__dict__', '__weakref__', '_time_synchronizer', '_last_update',
                 'band_id', '_spread', '_spotmult', '_prob_of_default_ac', '_recovery_rate',
                 'tdv_prob_of_default_ac', 'tdv_recovery_rate', 'tdv_spread', 'tdv_spotmult',)

    def __init__(
        self,
        band_id: str,
        *,
        model_engine: ProjModelEngine | None = None,
        tdv_spread_term_dim: list[int] | None = None,
        tdv_spotmult_term_dim: list[int] | None = None,
    ) -> None:
        """
        Initialize a CreditRisk object.

        Args:
            band_id (str): credit band identifier.
        """
        self.band_id: str = band_id
        self._spread: float | npt.NDArray[np.float64] | None = None
        self._spotmult: float | npt.NDArray[np.float64] | None = None
        self._prob_of_default_ac: float | None = None
        self._recovery_rate: float | None = None
        self._last_update: int | None = None

        create_tdv = lambda name: TDimVariable(name, model_engine=model_engine, owner=band_id, group='credit')
        self.tdv_prob_of_default_ac: TDimVariable = create_tdv("prob_of_default_ac")
        self.tdv_recovery_rate: TDimVariable = create_tdv("recovery_rate")
        if tdv_spread_term_dim is None:
            self.tdv_spread: TDimVariable = create_tdv("credit_spread")
        else:  # has term structure
            self.tdv_spread: TDimVariable = TDimVariable("credit_spread", dims=[tdv_spread_term_dim],
                                                         model_engine=model_engine, owner=band_id, group='credit')
        if tdv_spotmult_term_dim is None:
            self.tdv_spotmult: TDimVariable = create_tdv("credit_spotmult")
        else: # has term structure
            self.tdv_spotmult: TDimVariable = TDimVariable("credit_spotmult", dims=[tdv_spotmult_term_dim],
                                                           model_engine=model_engine, owner=band_id, group='credit')

    @property
    def last_update(self) -> int | None:
        """int: Last update time index."""
        return self._last_update

    @property
    def credit_spread(self) -> float | npt.NDArray[np.float64] | None:
        """npt.NDArray[np.float64]: Credit spread(s) as at period end."""
        return self._spread

    @property
    def credit_spotmult(self) -> float | npt.NDArray[np.float64] | None:
        """npt.NDArray[np.float64]: Spot rate multipliers as at period end."""
        return self._spotmult

    @property
    def prob_of_default_ac(self) -> float | None:
        """float: Probability of default (annual compounding) in period."""
        return self._prob_of_default_ac

    @property
    def recovery_rate(self) -> float | None:
        """float: Recovery rate in period."""
        return self._recovery_rate

    def no_change_on_update(self) -> None:
        self._on_exit_update()

    def update(self, prop_of_default_ac: float, recovery_rate: float, credit_spotmult: float | npt.NDArray[np.float64],
               credit_spread: float | npt.NDArray[np.float64]) -> None:
        """
        Update the credit parameters for the current time step.

        Args:
            prop_of_default_ac (float): Probability of default (annual compounding).
            recovery_rate (float): Recovery rate.
            credit_spotmult (float | npt.NDArray[np.float64]): Spot rate multipliers.
            credit_spread (float | npt.NDArray[np.float64]): Credit spread(s).
        """
        self._prob_of_default_ac = prop_of_default_ac
        self._recovery_rate = recovery_rate
        self._spotmult = credit_spotmult
        self._spread = credit_spread
        self._on_exit_update()

    def _on_exit_update(self) -> None:
        t = self.time
        self.tdv_prob_of_default_ac[t] = self._prob_of_default_ac
        self.tdv_recovery_rate[t] = self._recovery_rate

        # tdv_spread
        arr_len = len(self._spread) if isinstance(self._spread, np.ndarray) else 0
        tdv_term_dim = [int(i) for i in self.tdv_spread.dims[0]] if self.tdv_spread.dims else None
        if arr_len == 0: # scalar
            if tdv_term_dim is None:
                self.tdv_spread[t] = self._spread  # most often seen case, store a scalar value
            else:
                self.tdv_spread[t] = np.full(len(tdv_term_dim), self._spread)
        else: # has term structure
            if tdv_term_dim is None:
                self.tdv_spread[t] = self._spread[min(arr_len, 12)]  # store first year value only
            else:
                self.tdv_spread[t] = np.array([0 if i > arr_len else self._spread[i] for i in tdv_term_dim])

        # tdv_spotmult
        arr_len = len(self._spotmult) if isinstance(self._spotmult, np.ndarray) else 0
        tdv_term_dim = [int(i) for i in self.tdv_spotmult.dims[0]] if self.tdv_spotmult.dims else None
        if arr_len == 0: # scalar
            if tdv_term_dim is None:
                self.tdv_spotmult[t] = self._spotmult  # most often seen case, store a scalar value
            else:
                self.tdv_spotmult[t] = np.full(len(tdv_term_dim), self._spotmult)
        else: # has term structure
            if tdv_term_dim is None:
                self.tdv_spotmult[t] = self._spotmult[min(arr_len, 12)]  # store first year value only
            else:
                self.tdv_spotmult[t] = np.array([0 if i > arr_len else self._spotmult[i] for i in tdv_term_dim])
        self._last_update = t

    def __str__(self) -> str:
        return f"{type(self).__name__} - '{self.band_id}'"